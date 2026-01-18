import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class InvalidDomainName(WorldWideDNSException):

    def __init__(self, http_code, driver=None):
        value = 'Invalid domain name'
        super().__init__(value, http_code, 410, driver)