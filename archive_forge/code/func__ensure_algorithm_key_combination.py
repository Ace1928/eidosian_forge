from abc import ABC, abstractmethod  # pylint: disable=no-name-in-module
from typing import Any, Optional, Type
import dns.rdataclass
import dns.rdatatype
from dns.dnssectypes import Algorithm
from dns.exception import AlgorithmKeyMismatch
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.dnskeybase import Flag
@classmethod
def _ensure_algorithm_key_combination(cls, key: DNSKEY) -> None:
    if key.algorithm != cls.algorithm:
        raise AlgorithmKeyMismatch