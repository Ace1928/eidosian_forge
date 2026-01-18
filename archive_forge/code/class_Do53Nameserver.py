from typing import Optional, Union
from urllib.parse import urlparse
import dns.asyncbackend
import dns.asyncquery
import dns.inet
import dns.message
import dns.query
class Do53Nameserver(AddressAndPortNameserver):

    def __init__(self, address: str, port: int=53):
        super().__init__(address, port)

    def kind(self):
        return 'Do53'

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        if max_size:
            response = dns.query.tcp(request, self.address, timeout=timeout, port=self.port, source=source, source_port=source_port, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)
        else:
            response = dns.query.udp(request, self.address, timeout=timeout, port=self.port, source=source, source_port=source_port, raise_on_truncation=True, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, ignore_errors=True, ignore_unexpected=True)
        return response

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        if max_size:
            response = await dns.asyncquery.tcp(request, self.address, timeout=timeout, port=self.port, source=source, source_port=source_port, backend=backend, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing)
        else:
            response = await dns.asyncquery.udp(request, self.address, timeout=timeout, port=self.port, source=source, source_port=source_port, raise_on_truncation=True, backend=backend, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, ignore_errors=True, ignore_unexpected=True)
        return response