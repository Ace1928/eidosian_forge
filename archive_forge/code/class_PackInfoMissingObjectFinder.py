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
class PackInfoMissingObjectFinder(GreenThreadsMissingObjectFinder):

    def next(self):
        while True:
            if not self.objects_to_send:
                return None
            sha, name, leaf = self.objects_to_send.pop()
            if sha not in self.sha_done:
                break
        if not leaf:
            info = self.object_store.pack_info_get(sha)
            if info[0] == Commit.type_num:
                self.add_todo([(info[2], '', False)])
            elif info[0] == Tree.type_num:
                self.add_todo([tuple(i) for i in info[1]])
            elif info[0] == Tag.type_num:
                self.add_todo([(info[1], None, False)])
            if sha in self._tagged:
                self.add_todo([(self._tagged[sha], None, True)])
        self.sha_done.add(sha)
        self.progress('counting objects: %d\r' % len(self.sha_done))
        return (sha, name)