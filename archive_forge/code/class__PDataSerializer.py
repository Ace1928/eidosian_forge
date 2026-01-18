from __future__ import annotations
import collections.abc as cabc
import json
import typing as t
from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer
class _PDataSerializer(t.Protocol[_TSerialized]):

    def loads(self, payload: _TSerialized, /) -> t.Any:
        ...

    def dumps(self, obj: t.Any, /) -> _TSerialized:
        ...