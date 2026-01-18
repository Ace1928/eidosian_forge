from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from containerregistry.client import docker_creds
class Bearer(docker_creds.SchemeProvider):
    """Implementation for providing a transaction's Bearer token as creds."""

    def __init__(self, bearer_token):
        super(Bearer, self).__init__('Bearer')
        self._bearer_token = bearer_token

    @property
    def suffix(self):
        return self._bearer_token