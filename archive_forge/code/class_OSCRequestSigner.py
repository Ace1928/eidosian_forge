import hmac
import hashlib
from datetime import datetime
from libcloud.utils.py3 import urlquote
class OSCRequestSigner:
    """
    Class which handles signing the outgoing AWS requests.
    """

    def __init__(self, access_key: str, access_secret: str, version: str, connection):
        """
        :param access_key: Access key.
        :type access_key: ``str``

        :param access_secret: Access secret.
        :type access_secret: ``str``

        :param version: API version.
        :type version: ``str``

        :param connection: Connection instance.
        :type connection: :class:`Connection`
        """
        self.access_key = access_key
        self.access_secret = access_secret
        self.version = version
        self.connection = connection