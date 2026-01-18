import re
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, Tuple, Union
from ._abnf import method, request_target
from ._headers import Headers, normalize_and_validate
from ._util import bytesify, LocalProtocolError, validate
@dataclass(init=False, frozen=True)
class EndOfMessage(Event):
    """The end of an HTTP message.

    Fields:

    .. attribute:: headers

       Default value: ``[]``

       Any trailing headers attached to this message, represented as a list of
       (name, value) pairs. See :ref:`the header normalization rules
       <headers-format>` for details.

       Must be empty unless ``Transfer-Encoding: chunked`` is in use.

    """
    __slots__ = ('headers',)
    headers: Headers

    def __init__(self, *, headers: Union[Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]], None]=None, _parsed: bool=False) -> None:
        super().__init__()
        if headers is None:
            headers = Headers([])
        elif not isinstance(headers, Headers):
            headers = normalize_and_validate(headers, _parsed=_parsed)
        object.__setattr__(self, 'headers', headers)
    __hash__ = None