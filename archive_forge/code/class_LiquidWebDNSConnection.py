from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.liquidweb import APIException, LiquidWebResponse, LiquidWebConnection
class LiquidWebDNSConnection(LiquidWebConnection):
    responseCls = LiquidWebDNSResponse