import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class WorldWideDNSException(ProviderError):

    def __init__(self, value, http_code, code, driver=None):
        self.code = code
        super().__init__(value, http_code, driver)