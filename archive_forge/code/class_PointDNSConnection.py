import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
class PointDNSConnection(ConnectionUserAndKey):
    host = 'pointhq.com'
    responseCls = PointDNSDNSResponse

    def add_default_headers(self, headers):
        """
        Add headers that are necessary for every request

        This method adds ``token`` to the request.
        """
        b64string = b('{}:{}'.format(self.user_id, self.key))
        token = base64.b64encode(b64string)
        headers['Authorization'] = 'Basic %s' % token
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        return headers