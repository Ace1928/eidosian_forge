import json
import os
import posixpath
import stat
import sys
import tempfile
import urllib.parse as urlparse
import zlib
from configparser import ConfigParser
from io import BytesIO
from geventhttpclient import HTTPClient
from ..greenthreads import GreenThreadsMissingObjectFinder
from ..lru_cache import LRUSizeCache
from ..object_store import INFODIR, PACKDIR, PackBasedObjectStore
from ..objects import S_ISGITLINK, Blob, Commit, Tag, Tree
from ..pack import (
from ..protocol import TCP_GIT_PORT
from ..refs import InfoRefsContainer, read_info_refs, write_info_refs
from ..repo import OBJECTDIR, BaseRepo
from ..server import Backend, TCPGitServer
def _collect_ancestors(self, heads, common=set()):

    def _find_parents(commit):
        for pack in self.packs:
            if commit in pack:
                try:
                    parents = pack.pack_info[commit][1]
                except KeyError:
                    return []
                return parents
    bases = set()
    commits = set()
    queue = []
    queue.extend(heads)
    while queue:
        e = queue.pop(0)
        if e in common:
            bases.add(e)
        elif e not in commits:
            commits.add(e)
            parents = _find_parents(e)
            queue.extend(parents)
    return (commits, bases)