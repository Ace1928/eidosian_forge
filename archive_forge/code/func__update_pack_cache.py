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
def _update_pack_cache(self):
    objects = self.scon.get_container_objects()
    pack_files = [o['name'].replace('.pack', '') for o in objects if o['name'].endswith('.pack')]
    ret = []
    for basename in pack_files:
        pack = SwiftPack(basename, scon=self.scon)
        self._pack_cache[basename] = pack
        ret.append(pack)
    return ret