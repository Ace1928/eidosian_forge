import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class CouldntReachNameServer(WorldWideDNSException):

    def __init__(self, http_code, driver=None):
        value = "Couldn't reach the name server, try again later"
        super().__init__(value, http_code, 450, driver)