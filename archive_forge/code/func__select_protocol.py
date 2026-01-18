import logging
import sys
import weakref
import webob
from wsme.exc import ClientSideError, UnknownFunction
from wsme.protocol import getprotocol
from wsme.rest import scan_api
import wsme.api
import wsme.types
def _select_protocol(self, request):
    log.debug('Selecting a protocol for the following request :\nheaders: %s\nbody: %s', request.headers.items(), request.content_length and (request.content_length > 512 and request.body[:512] or request.body) or '')
    protocol = None
    error = ClientSideError(status_code=406)
    path = str(request.path)
    assert path.startswith(self._webpath)
    path = path[len(self._webpath) + 1:]
    if 'wsmeproto' in request.params:
        return self._get_protocol(request.params['wsmeproto'])
    else:
        for p in self.protocols:
            try:
                if p.accept(request):
                    protocol = p
                    break
            except ClientSideError as e:
                error = e
        if not protocol:
            raise error
    return protocol