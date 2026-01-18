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
class SwiftInfoRefsContainer(InfoRefsContainer):
    """Manage references in info/refs object."""

    def __init__(self, scon, store) -> None:
        self.scon = scon
        self.filename = 'info/refs'
        self.store = store
        f = self.scon.get_object(self.filename)
        if not f:
            f = BytesIO(b'')
        super().__init__(f)

    def _load_check_ref(self, name, old_ref):
        self._check_refname(name)
        f = self.scon.get_object(self.filename)
        if not f:
            return {}
        refs = read_info_refs(f)
        if old_ref is not None:
            if refs[name] != old_ref:
                return False
        return refs

    def _write_refs(self, refs):
        f = BytesIO()
        f.writelines(write_info_refs(refs, self.store))
        self.scon.put_object(self.filename, f)

    def set_if_equals(self, name, old_ref, new_ref):
        """Set a refname to new_ref only if it currently equals old_ref."""
        if name == 'HEAD':
            return True
        refs = self._load_check_ref(name, old_ref)
        if not isinstance(refs, dict):
            return False
        refs[name] = new_ref
        self._write_refs(refs)
        self._refs[name] = new_ref
        return True

    def remove_if_equals(self, name, old_ref):
        """Remove a refname only if it currently equals old_ref."""
        if name == 'HEAD':
            return True
        refs = self._load_check_ref(name, old_ref)
        if not isinstance(refs, dict):
            return False
        del refs[name]
        self._write_refs(refs)
        del self._refs[name]
        return True

    def allkeys(self):
        try:
            self._refs['HEAD'] = self._refs['refs/heads/master']
        except KeyError:
            pass
        return self._refs.keys()