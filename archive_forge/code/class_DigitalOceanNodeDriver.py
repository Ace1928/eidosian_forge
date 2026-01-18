import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
class DigitalOceanNodeDriver(NodeDriver):
    """
    DigitalOcean NodeDriver defaulting to using APIv2.

    :keyword    key: Personal Access Token required for authentication.
    :type       key: ``str``

    :keyword    secret: Previously used with API version ``v1``. (deprecated)
    :type       secret: ``str``

    :keyword    api_version: Specifies the API version to use. Defaults to
                             using ``v2``, currently the only valid option.
                             (optional)
    :type       api_version: ``str``
    """
    type = Provider.DIGITAL_OCEAN
    name = 'DigitalOcean'
    website = 'https://www.digitalocean.com'

    def __new__(cls, key, secret=None, api_version='v2', **kwargs):
        if cls is DigitalOceanNodeDriver:
            if api_version == 'v1' or secret is not None:
                if secret is not None and api_version == 'v2':
                    raise InvalidCredsError('secret not accepted for v2 authentication')
                raise DigitalOcean_v1_Error()
            elif api_version == 'v2':
                cls = DigitalOcean_v2_NodeDriver
            else:
                raise NotImplementedError('Unsupported API version: %s' % api_version)
        return super().__new__(cls, **kwargs)