import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class DomainBanned(WorldWideDNSException):

    def __init__(self, http_code, driver=None):
        value = 'Domain is listed in DNSBL and is banned from our servers'
        super().__init__(value, http_code, 409, driver)