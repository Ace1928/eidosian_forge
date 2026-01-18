from __future__ import annotations
import collections.abc as cabc
import json
import typing as t
from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer
def _loads_unsafe_impl(self, s: str | bytes, salt: str | bytes | None, load_kwargs: dict[str, t.Any] | None=None, load_payload_kwargs: dict[str, t.Any] | None=None) -> tuple[bool, t.Any]:
    """Low level helper function to implement :meth:`loads_unsafe`
        in serializer subclasses.
        """
    if load_kwargs is None:
        load_kwargs = {}
    try:
        return (True, self.loads(s, salt=salt, **load_kwargs))
    except BadSignature as e:
        if e.payload is None:
            return (False, None)
        if load_payload_kwargs is None:
            load_payload_kwargs = {}
        try:
            return (False, self.load_payload(e.payload, **load_payload_kwargs))
        except BadPayload:
            return (False, None)