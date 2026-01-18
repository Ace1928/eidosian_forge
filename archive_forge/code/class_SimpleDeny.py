import base64
import contextlib
import functools
import hashlib
import struct
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast
import dns._features
import dns.exception
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.transaction
import dns.zone
from dns.dnssectypes import Algorithm, DSDigest, NSEC3Hash
from dns.exception import (  # pylint: disable=W0611
from dns.rdtypes.ANY.CDNSKEY import CDNSKEY
from dns.rdtypes.ANY.CDS import CDS
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.ANY.DS import DS
from dns.rdtypes.ANY.NSEC import NSEC, Bitmap
from dns.rdtypes.ANY.NSEC3PARAM import NSEC3PARAM
from dns.rdtypes.ANY.RRSIG import RRSIG, sigtime_to_posixtime
from dns.rdtypes.dnskeybase import Flag
class SimpleDeny(Policy):

    def __init__(self, deny_sign, deny_validate, deny_create_ds, deny_validate_ds):
        super().__init__()
        self._deny_sign = deny_sign
        self._deny_validate = deny_validate
        self._deny_create_ds = deny_create_ds
        self._deny_validate_ds = deny_validate_ds

    def ok_to_sign(self, key: DNSKEY) -> bool:
        return key.algorithm not in self._deny_sign

    def ok_to_validate(self, key: DNSKEY) -> bool:
        return key.algorithm not in self._deny_validate

    def ok_to_create_ds(self, algorithm: DSDigest) -> bool:
        return algorithm not in self._deny_create_ds

    def ok_to_validate_ds(self, algorithm: DSDigest) -> bool:
        return algorithm not in self._deny_validate_ds