from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.gandi import BaseObject
from libcloud.common.types import LibcloudError, InvalidCredsError
class LinodeConnection(ConnectionKey):
    """
    A connection to the Linode API

    Wraps SSL connections to the Linode API, automagically injecting the
    parameters that the API needs for each request.
    """
    host = API_HOST
    responseCls = LinodeResponse

    def add_default_params(self, params):
        """
        Add parameters that are necessary for every request

        This method adds ``api_key`` and ``api_responseFormat`` to
        the request.
        """
        params['api_key'] = self.key
        params['api_responseFormat'] = 'json'
        return params