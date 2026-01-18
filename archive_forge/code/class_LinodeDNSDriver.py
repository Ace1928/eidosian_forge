from datetime import datetime
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.linode import (
class LinodeDNSDriver(DNSDriver):
    type = Provider.LINODE
    name = 'Linode DNS'
    website = 'http://www.linode.com/'

    def __new__(cls, key, secret=None, secure=True, host=None, port=None, api_version=DEFAULT_API_VERSION, **kwargs):
        if cls is LinodeDNSDriver:
            if api_version == '3.0':
                cls = LinodeDNSDriverV3
            elif api_version == '4.0':
                cls = LinodeDNSDriverV4
            else:
                raise NotImplementedError('No Linode driver found for API version: %s' % api_version)
        return super().__new__(cls)