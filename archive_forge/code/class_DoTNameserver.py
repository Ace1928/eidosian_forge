from typing import Optional, Union
from urllib.parse import urlparse
import dns.asyncbackend
import dns.asyncquery
import dns.inet
import dns.message
import dns.query
class DoTNameserver(AddressAndPortNameserver):

    def __init__(self, address: str, port: int=853, hostname: Optional[str]=None, verify: Union[bool, str]=True):
        super().__init__(address, port)
        self.hostname = hostname
        self.verify = verify

    def kind(self):
        return 'DoT'

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool=False, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return dns.query.tls(request, self.address, port=self.port, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, server_hostname=self.hostname, verify=self.verify)

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return await dns.asyncquery.tls(request, self.address, port=self.port, timeout=timeout, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, server_hostname=self.hostname, verify=self.verify)