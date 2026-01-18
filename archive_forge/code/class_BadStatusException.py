from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
import httplib2
class BadStatusException(Exception):
    """Exceptions when an unexpected HTTP status is returned."""

    def __init__(self, resp, content):
        message = 'Response:\n{resp}\nContent:\n{content}'.format(resp=resp, content=content)
        super(BadStatusException, self).__init__(message)
        self._resp = resp
        self._content = content

    @property
    def resp(self):
        return self._resp

    @property
    def status(self):
        return self._resp.status

    @property
    def content(self):
        return self._content