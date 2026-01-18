from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
class DNSimpleDNSConnection(ConnectionUserAndKey):
    host = 'api.dnsimple.com'
    responseCls = DNSimpleDNSResponse

    def add_default_headers(self, headers):
        """
        Add headers that are necessary for every request

        This method adds ``token`` to the request.
        """
        headers['X-DNSimple-Token'] = '{}:{}'.format(self.user_id, self.key)
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        return headers