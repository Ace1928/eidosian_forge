from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.zonomi import ZonomiResponse, ZonomiException, ZonomiConnection
class ZonomiDNSResponse(ZonomiResponse):
    pass