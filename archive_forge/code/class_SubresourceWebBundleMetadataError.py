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
@event_class('Network.subresourceWebBundleMetadataError')
@dataclass
class SubresourceWebBundleMetadataError:
    """
    **EXPERIMENTAL**

    Fired once when parsing the .wbn file has failed.
    """
    request_id: RequestId
    error_message: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleMetadataError:
        return cls(request_id=RequestId.from_json(json['requestId']), error_message=str(json['errorMessage']))