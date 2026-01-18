import json
from typing import Any, Dict, List, Optional
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import urlencode
from libcloud.common.vultr import (
class VultrDNSDriver(DNSDriver):
    type = Provider.VULTR
    name = 'Vultr DNS'
    website = 'https://www.vultr.com'

    def __new__(cls, key, secret=None, secure=True, host=None, port=None, api_version=DEFAULT_API_VERSION, region=None, **kwargs):
        if cls is VultrDNSDriver:
            if api_version == '1':
                cls = VultrDNSDriverV1
            elif api_version == '2':
                cls = VultrDNSDriverV2
            else:
                raise NotImplementedError('No Vultr driver found for API version: %s' % api_version)
        return super().__new__(cls)