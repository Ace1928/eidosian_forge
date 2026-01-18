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
class LoadNetworkResourcePageResult:
    """
    An object providing the result of a network resource load.
    """
    success: bool
    net_error: typing.Optional[float] = None
    net_error_name: typing.Optional[str] = None
    http_status_code: typing.Optional[float] = None
    stream: typing.Optional[io.StreamHandle] = None
    headers: typing.Optional[network.Headers] = None

    def to_json(self):
        json = dict()
        json['success'] = self.success
        if self.net_error is not None:
            json['netError'] = self.net_error
        if self.net_error_name is not None:
            json['netErrorName'] = self.net_error_name
        if self.http_status_code is not None:
            json['httpStatusCode'] = self.http_status_code
        if self.stream is not None:
            json['stream'] = self.stream.to_json()
        if self.headers is not None:
            json['headers'] = self.headers.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(success=bool(json['success']), net_error=float(json['netError']) if 'netError' in json else None, net_error_name=str(json['netErrorName']) if 'netErrorName' in json else None, http_status_code=float(json['httpStatusCode']) if 'httpStatusCode' in json else None, stream=io.StreamHandle.from_json(json['stream']) if 'stream' in json else None, headers=network.Headers.from_json(json['headers']) if 'headers' in json else None)