from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.luadns import LuadnsResponse, LuadnsException, LuadnsConnection
class LuadnsDNSResponse(LuadnsResponse):
    pass