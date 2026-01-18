import base64
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class LiquidWebException(ProviderError):
    """The base class for other Liquidweb exceptions"""

    def __init__(self, value, http_code, extra=None):
        """
        :param value: message contained in error
        :type  value: ``str``

        :param http_code: error code
        :type  http_code: ``int``

        :param extra: extra fields specific to error type
        :type  extra: ``list``
        """
        self.extra = extra
        super().__init__(value, http_code, driver=None)

    def __str__(self):
        return '{}  {}'.format(self.http_code, self.value)

    def __repr__(self):
        return 'LiquidWebException {} {}'.format(self.http_code, self.value)