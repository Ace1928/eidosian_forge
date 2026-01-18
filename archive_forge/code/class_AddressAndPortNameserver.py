from typing import Optional, Union
from urllib.parse import urlparse
import dns.asyncbackend
import dns.asyncquery
import dns.inet
import dns.message
import dns.query
class AddressAndPortNameserver(Nameserver):

    def __init__(self, address: str, port: int):
        super().__init__()
        self.address = address
        self.port = port

    def kind(self) -> str:
        raise NotImplementedError

    def is_always_max_size(self) -> bool:
        return False

    def __str__(self):
        ns_kind = self.kind()
        return f'{ns_kind}:{self.address}@{self.port}'

    def answer_nameserver(self) -> str:
        return self.address

    def answer_port(self) -> int:
        return self.port