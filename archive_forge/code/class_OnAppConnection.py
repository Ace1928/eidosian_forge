from base64 import b64encode
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
class OnAppConnection(ConnectionUserAndKey):
    """
    OnApp connection class
    """
    responseCls = OnAppResponse

    def add_default_headers(self, headers):
        """
        Add Basic Authentication header to all the requests.
        It injects the "Authorization: Basic Base64String===" header
        in each request

        :type  headers: ``dict``
        :param headers: Default input headers

        :rtype:         ``dict``
        :return:        Default input headers with the "Authorization" header.
        """
        b64string = b('{}:{}'.format(self.user_id, self.key))
        encoded = b64encode(b64string).decode('utf-8')
        headers['Authorization'] = 'Basic ' + encoded
        return headers