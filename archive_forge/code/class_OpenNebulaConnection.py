import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebulaConnection(ConnectionUserAndKey):
    """
    Connection class for the OpenNebula.org driver.
    with plain_auth support
    """
    host = API_HOST
    port = API_PORT
    secure = API_SECURE
    plain_auth = API_PLAIN_AUTH
    responseCls = OpenNebulaResponse

    def __init__(self, *args, **kwargs):
        if 'plain_auth' in kwargs:
            self.plain_auth = kwargs.pop('plain_auth')
        super().__init__(*args, **kwargs)

    def add_default_headers(self, headers):
        """
        Add headers required by the OpenNebula.org OCCI interface.

        Includes adding Basic HTTP Authorization headers for authenticating
        against the OpenNebula.org OCCI interface.

        :type  headers: ``dict``
        :param headers: Dictionary containing HTTP headers.

        :rtype:  ``dict``
        :return: Dictionary containing updated headers.
        """
        if self.plain_auth:
            passwd = self.key
        else:
            passwd = hashlib.sha1(b(self.key)).hexdigest()
        headers['Authorization'] = 'Basic %s' % b64encode(b('{}:{}'.format(self.user_id, passwd))).decode('utf-8')
        return headers