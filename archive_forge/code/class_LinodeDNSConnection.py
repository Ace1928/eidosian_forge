from datetime import datetime
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.linode import (
class LinodeDNSConnection(LinodeConnection):
    responseCls = LinodeDNSResponse