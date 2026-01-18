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
class DictBackend(Backend):
    """Trivial backend that looks up Git repositories in a dictionary."""

    def __init__(self, repos) -> None:
        self.repos = repos

    def open_repository(self, path: str) -> BaseRepo:
        logger.debug('Opening repository at %s', path)
        try:
            return self.repos[path]
        except KeyError as exc:
            raise NotGitRepository('No git repository was found at {path}'.format(**dict(path=path))) from exc