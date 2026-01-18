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
class BackendRepo(TypingProtocol):
    """Repository abstraction used by the Git server.

    The methods required here are a subset of those provided by
    dulwich.repo.Repo.
    """
    object_store: PackedObjectContainer
    refs: RefsContainer

    def get_refs(self) -> Dict[bytes, bytes]:
        """Get all the refs in the repository.

        Returns: dict of name -> sha
        """
        raise NotImplementedError

    def get_peeled(self, name: bytes) -> Optional[bytes]:
        """Return the cached peeled value of a ref, if available.

        Args:
          name: Name of the ref to peel
        Returns: The peeled value of the ref. If the ref is known not point to
            a tag, this will be the SHA the ref refers to. If no cached
            information about a tag is available, this method may return None,
            but it should attempt to peel the tag if possible.
        """
        return None

    def find_missing_objects(self, determine_wants, graph_walker, progress, get_tagged=None):
        """Yield the objects required for a list of commits.

        Args:
          progress: is a callback to send progress messages to the client
          get_tagged: Function that returns a dict of pointed-to sha ->
            tag sha for including tags.
        """
        raise NotImplementedError