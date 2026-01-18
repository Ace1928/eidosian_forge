from typing import Optional, Union
from urllib.parse import urlparse
import dns.asyncbackend
import dns.asyncquery
import dns.inet
import dns.message
import dns.query
class DoHNameserver(Nameserver):

    def __init__(self, url: str, bootstrap_address: Optional[str]=None, verify: Union[bool, str]=True, want_get: bool=False):
        super().__init__()
        self.url = url
        self.bootstrap_address = bootstrap_address
        self.verify = verify
        self.want_get = want_get

    def kind(self):
        return 'DoH'

    def is_always_max_size(self) -> bool:
        return True

    def __str__(self):
        return self.url

    def answer_nameserver(self) -> str:
        return self.url

    def answer_port(self) -> int:
        port = urlparse(self.url).port
        if port is None:
            port = 443
        return port

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool=False, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return dns.query.https(request, self.url, timeout=timeout, source=source, source_port=source_port, bootstrap_address=self.bootstrap_address, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, verify=self.verify, post=not self.want_get)

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        return await dns.asyncquery.https(request, self.url, timeout=timeout, source=source, source_port=source_port, bootstrap_address=self.bootstrap_address, one_rr_per_rrset=one_rr_per_rrset, ignore_trailing=ignore_trailing, verify=self.verify, post=not self.want_get)