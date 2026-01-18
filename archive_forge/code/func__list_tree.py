import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _list_tree(self, relpath, depth):
    abspath = self._remote_path(relpath)
    propfind = b'<?xml version="1.0" encoding="utf-8" ?>\n   <D:propfind xmlns:D="DAV:">\n     <D:allprop/>\n   </D:propfind>\n'
    response = self.request('PROPFIND', abspath, body=propfind, headers={'Depth': '{}'.format(depth), 'Content-Type': 'application/xml; charset="utf-8"'})
    code = response.status
    if code == 404:
        raise transport.NoSuchFile(abspath)
    if code == 409:
        raise transport.NoSuchFile(abspath)
    if code != 207:
        self._raise_http_error(abspath, response, 'unable to list  %r directory' % abspath)
    return _extract_dir_content(abspath, response)