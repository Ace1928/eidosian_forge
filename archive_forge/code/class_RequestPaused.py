from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
from . import network
from . import page
@event_class('Fetch.requestPaused')
@dataclass
class RequestPaused:
    """
    Issued when the domain is enabled and the request URL matches the
    specified filter. The request is paused until the client responds
    with one of continueRequest, failRequest or fulfillRequest.
    The stage of the request can be determined by presence of responseErrorReason
    and responseStatusCode -- the request is at the response stage if either
    of these fields is present and in the request stage otherwise.
    """
    request_id: RequestId
    request: network.Request
    frame_id: page.FrameId
    resource_type: network.ResourceType
    response_error_reason: typing.Optional[network.ErrorReason]
    response_status_code: typing.Optional[int]
    response_headers: typing.Optional[typing.List[HeaderEntry]]
    network_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestPaused:
        return cls(request_id=RequestId.from_json(json['requestId']), request=network.Request.from_json(json['request']), frame_id=page.FrameId.from_json(json['frameId']), resource_type=network.ResourceType.from_json(json['resourceType']), response_error_reason=network.ErrorReason.from_json(json['responseErrorReason']) if 'responseErrorReason' in json else None, response_status_code=int(json['responseStatusCode']) if 'responseStatusCode' in json else None, response_headers=[HeaderEntry.from_json(i) for i in json['responseHeaders']] if 'responseHeaders' in json else None, network_id=RequestId.from_json(json['networkId']) if 'networkId' in json else None)