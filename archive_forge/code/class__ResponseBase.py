import re
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, Tuple, Union
from ._abnf import method, request_target
from ._headers import Headers, normalize_and_validate
from ._util import bytesify, LocalProtocolError, validate
@dataclass(init=False, frozen=True)
class _ResponseBase(Event):
    __slots__ = ('headers', 'http_version', 'reason', 'status_code')
    headers: Headers
    http_version: bytes
    reason: bytes
    status_code: int

    def __init__(self, *, headers: Union[Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]]], status_code: int, http_version: Union[bytes, str]=b'1.1', reason: Union[bytes, str]=b'', _parsed: bool=False) -> None:
        super().__init__()
        if isinstance(headers, Headers):
            object.__setattr__(self, 'headers', headers)
        else:
            object.__setattr__(self, 'headers', normalize_and_validate(headers, _parsed=_parsed))
        if not _parsed:
            object.__setattr__(self, 'reason', bytesify(reason))
            object.__setattr__(self, 'http_version', bytesify(http_version))
            if not isinstance(status_code, int):
                raise LocalProtocolError('status code must be integer')
            object.__setattr__(self, 'status_code', int(status_code))
        else:
            object.__setattr__(self, 'reason', reason)
            object.__setattr__(self, 'http_version', http_version)
            object.__setattr__(self, 'status_code', status_code)
        self.__post_init__()

    def __post_init__(self) -> None:
        pass
    __hash__ = None