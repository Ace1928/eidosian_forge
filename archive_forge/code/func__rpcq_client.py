from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
@contextmanager
def _rpcq_client(self) -> Iterator[rpcq.Client]:
    client = rpcq.Client(endpoint=self.base_url, timeout=self.timeout)
    try:
        yield client
    finally:
        client.close()