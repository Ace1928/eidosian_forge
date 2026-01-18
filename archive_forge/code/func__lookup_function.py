import logging
import sys
import weakref
import webob
from wsme.exc import ClientSideError, UnknownFunction
from wsme.protocol import getprotocol
from wsme.rest import scan_api
import wsme.api
import wsme.types
def _lookup_function(self, path):
    if not self._api:
        self.getapi()
    for fpath, f, fdef, args in self._api:
        if path == fpath:
            return (f, fdef, args)
    raise UnknownFunction('/'.join(path))