from xml.etree.ElementTree import tostring
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib, ensure_string
from libcloud.common.durabledns import (
from libcloud.common.durabledns import _schema_builder as api_schema_builder
class DurableDNSResponse(DurableResponse):
    pass