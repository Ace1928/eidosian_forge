import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
class FileSystemBackend(Backend):
    """Simple backend looking up Git repositories in the local file system."""

    def __init__(self, root=os.sep) -> None:
        super().__init__()
        self.root = (os.path.abspath(root) + os.sep).replace(os.sep * 2, os.sep)

    def open_repository(self, path):
        logger.debug('opening repository at %s', path)
        abspath = os.path.abspath(os.path.join(self.root, path)) + os.sep
        normcase_abspath = os.path.normcase(abspath)
        normcase_root = os.path.normcase(self.root)
        if not normcase_abspath.startswith(normcase_root):
            raise NotGitRepository(f'Path {path!r} not inside root {self.root!r}')
        return Repo(abspath)