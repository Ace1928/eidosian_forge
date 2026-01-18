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
@dataclass
class SignedExchangeSignature:
    """
    Information about a signed exchange signature.
    https://wicg.github.io/webpackage/draft-yasskin-httpbis-origin-signed-exchanges-impl.html#rfc.section.3.1
    """
    label: str
    signature: str
    integrity: str
    validity_url: str
    date: int
    expires: int
    cert_url: typing.Optional[str] = None
    cert_sha256: typing.Optional[str] = None
    certificates: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['label'] = self.label
        json['signature'] = self.signature
        json['integrity'] = self.integrity
        json['validityUrl'] = self.validity_url
        json['date'] = self.date
        json['expires'] = self.expires
        if self.cert_url is not None:
            json['certUrl'] = self.cert_url
        if self.cert_sha256 is not None:
            json['certSha256'] = self.cert_sha256
        if self.certificates is not None:
            json['certificates'] = [i for i in self.certificates]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(label=str(json['label']), signature=str(json['signature']), integrity=str(json['integrity']), validity_url=str(json['validityUrl']), date=int(json['date']), expires=int(json['expires']), cert_url=str(json['certUrl']) if 'certUrl' in json else None, cert_sha256=str(json['certSha256']) if 'certSha256' in json else None, certificates=[str(i) for i in json['certificates']] if 'certificates' in json else None)