from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import re
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_creds as v2_2_creds
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
class V2DiagnosticException(Exception):
    """Exceptions when an unexpected HTTP status is returned."""

    def __init__(self, resp, content):
        self._resp = resp
        self._diagnostics = _DiagnosticsFromContent(content)
        message = '\n'.join(['response: %s' % resp] + ['%s: %s' % (d.message, d.detail) for d in self._diagnostics])
        super(V2DiagnosticException, self).__init__(message)

    @property
    def diagnostics(self):
        return self._diagnostics

    @property
    def response(self):
        return self._resp

    @property
    def status(self):
        return self._resp.status