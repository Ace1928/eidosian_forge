import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider
class RancherConnection(ConnectionUserAndKey):
    responseCls = RancherResponse
    timeout = 30

    def add_default_headers(self, headers):
        """
        Add parameters that are necessary for every request
        If user and password are specified, include a base http auth
        header
        """
        headers['Content-Type'] = 'application/json'
        headers['Accept'] = 'application/json'
        if self.key and self.user_id:
            user_b64 = base64.b64encode(b('{}:{}'.format(self.user_id, self.key)))
            headers['Authorization'] = 'Basic %s' % user_b64.decode('utf-8')
        return headers