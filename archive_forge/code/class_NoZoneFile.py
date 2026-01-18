import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class NoZoneFile(WorldWideDNSException):

    def __init__(self, http_code, driver=None):
        value = 'No zone file in the name server queried'
        super().__init__(value, http_code, 451, driver)