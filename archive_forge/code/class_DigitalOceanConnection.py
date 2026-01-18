from libcloud.utils.py3 import httplib, parse_qs, urlparse
from libcloud.common.base import BaseDriver, JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError
class DigitalOceanConnection(DigitalOcean_v2_Connection):
    """
    Connection class for the DigitalOcean driver.
    """
    pass