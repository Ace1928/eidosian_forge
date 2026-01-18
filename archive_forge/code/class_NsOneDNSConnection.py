from libcloud.dns.base import Zone, Record, DNSDriver, RecordType
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nsone import NsOneResponse, NsOneException, NsOneConnection
class NsOneDNSConnection(NsOneConnection):
    responseCls = NsOneDNSResponse