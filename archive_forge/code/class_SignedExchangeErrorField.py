from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
class SignedExchangeErrorField(enum.Enum):
    """
    Field type for a signed exchange related error.
    """
    SIGNATURE_SIG = 'signatureSig'
    SIGNATURE_INTEGRITY = 'signatureIntegrity'
    SIGNATURE_CERT_URL = 'signatureCertUrl'
    SIGNATURE_CERT_SHA256 = 'signatureCertSha256'
    SIGNATURE_VALIDITY_URL = 'signatureValidityUrl'
    SIGNATURE_TIMESTAMPS = 'signatureTimestamps'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)