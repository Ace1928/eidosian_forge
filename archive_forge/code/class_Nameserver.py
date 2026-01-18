from typing import Optional, Union
from urllib.parse import urlparse
import dns.asyncbackend
import dns.asyncquery
import dns.inet
import dns.message
import dns.query
class Nameserver:

    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError

    def kind(self) -> str:
        raise NotImplementedError

    def is_always_max_size(self) -> bool:
        raise NotImplementedError

    def answer_nameserver(self) -> str:
        raise NotImplementedError

    def answer_port(self) -> int:
        raise NotImplementedError

    def query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        raise NotImplementedError

    async def async_query(self, request: dns.message.QueryMessage, timeout: float, source: Optional[str], source_port: int, max_size: bool, backend: dns.asyncbackend.Backend, one_rr_per_rrset: bool=False, ignore_trailing: bool=False) -> dns.message.Message:
        raise NotImplementedError