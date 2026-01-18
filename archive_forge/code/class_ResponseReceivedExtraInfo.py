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
@event_class('Network.responseReceivedExtraInfo')
@dataclass
class ResponseReceivedExtraInfo:
    """
    **EXPERIMENTAL**

    Fired when additional information about a responseReceived event is available from the network
    stack. Not every responseReceived event will have an additional responseReceivedExtraInfo for
    it, and responseReceivedExtraInfo may be fired before or after responseReceived.
    """
    request_id: RequestId
    blocked_cookies: typing.List[BlockedSetCookieWithReason]
    headers: Headers
    resource_ip_address_space: IPAddressSpace
    status_code: int
    headers_text: typing.Optional[str]
    cookie_partition_key: typing.Optional[str]
    cookie_partition_key_opaque: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResponseReceivedExtraInfo:
        return cls(request_id=RequestId.from_json(json['requestId']), blocked_cookies=[BlockedSetCookieWithReason.from_json(i) for i in json['blockedCookies']], headers=Headers.from_json(json['headers']), resource_ip_address_space=IPAddressSpace.from_json(json['resourceIPAddressSpace']), status_code=int(json['statusCode']), headers_text=str(json['headersText']) if 'headersText' in json else None, cookie_partition_key=str(json['cookiePartitionKey']) if 'cookiePartitionKey' in json else None, cookie_partition_key_opaque=bool(json['cookiePartitionKeyOpaque']) if 'cookiePartitionKeyOpaque' in json else None)