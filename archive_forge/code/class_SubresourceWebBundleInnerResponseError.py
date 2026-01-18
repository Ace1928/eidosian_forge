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
@event_class('Network.subresourceWebBundleInnerResponseError')
@dataclass
class SubresourceWebBundleInnerResponseError:
    """
    **EXPERIMENTAL**

    Fired when request for resources within a .wbn file failed.
    """
    inner_request_id: RequestId
    inner_request_url: str
    error_message: str
    bundle_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleInnerResponseError:
        return cls(inner_request_id=RequestId.from_json(json['innerRequestId']), inner_request_url=str(json['innerRequestURL']), error_message=str(json['errorMessage']), bundle_request_id=RequestId.from_json(json['bundleRequestId']) if 'bundleRequestId' in json else None)