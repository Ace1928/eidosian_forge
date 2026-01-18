import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class ErrorOnReloadInNameServer(WorldWideDNSException):

    def __init__(self, server, http_code, driver=None):
        if server == 1:
            value = 'Name server #1 kicked an error on reload, contact support'
            code = 411
        elif server == 2:
            value = 'Name server #2 kicked an error on reload, contact support'
            code = 412
        elif server == 3:
            value = 'Name server #3 kicked an error on reload, contact support'
            code = 413
        super().__init__(value, http_code, code, driver)