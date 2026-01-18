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
@event_class('Network.subresourceWebBundleInnerResponseParsed')
@dataclass
class SubresourceWebBundleInnerResponseParsed:
    """
    **EXPERIMENTAL**

    Fired when handling requests for resources within a .wbn file.
    Note: this will only be fired for resources that are requested by the webpage.
    """
    inner_request_id: RequestId
    inner_request_url: str
    bundle_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleInnerResponseParsed:
        return cls(inner_request_id=RequestId.from_json(json['innerRequestId']), inner_request_url=str(json['innerRequestURL']), bundle_request_id=RequestId.from_json(json['bundleRequestId']) if 'bundleRequestId' in json else None)