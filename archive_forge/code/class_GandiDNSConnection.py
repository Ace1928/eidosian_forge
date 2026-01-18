from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi import GandiResponse, BaseGandiDriver, GandiConnection
class GandiDNSConnection(GandiConnection):
    responseCls = GandiDNSResponse