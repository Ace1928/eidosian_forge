from libcloud.utils.py3 import httplib, parse_qs, urlparse
from libcloud.common.base import BaseDriver, JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError
class DigitalOcean_v1_Error(LibcloudError):
    """
    Exception for when attempting to use version 1
    of the DigitalOcean API which is no longer
    supported.
    """

    def __init__(self, value='Driver no longer supported: Version 1 of the DigitalOcean API reached end of life on November 9, 2015. Use the v2 driver. Please visit: https://developers.digitalocean.com/documentation/changelog/api-v1/sunsetting-api-v1/', driver=None):
        super().__init__(value, driver=driver)