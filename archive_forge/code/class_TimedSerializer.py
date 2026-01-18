from __future__ import annotations
import collections.abc as cabc
import time
import typing as t
from datetime import datetime
from datetime import timezone
from .encoding import base64_decode
from .encoding import base64_encode
from .encoding import bytes_to_int
from .encoding import int_to_bytes
from .encoding import want_bytes
from .exc import BadSignature
from .exc import BadTimeSignature
from .exc import SignatureExpired
from .serializer import _TSerialized
from .serializer import Serializer
from .signer import Signer
class TimedSerializer(Serializer[_TSerialized]):
    """Uses :class:`TimestampSigner` instead of the default
    :class:`.Signer`.
    """
    default_signer: type[TimestampSigner] = TimestampSigner

    def iter_unsigners(self, salt: str | bytes | None=None) -> cabc.Iterator[TimestampSigner]:
        return t.cast('cabc.Iterator[TimestampSigner]', super().iter_unsigners(salt))

    def loads(self, s: str | bytes, max_age: int | None=None, return_timestamp: bool=False, salt: str | bytes | None=None) -> t.Any:
        """Reverse of :meth:`dumps`, raises :exc:`.BadSignature` if the
        signature validation fails. If a ``max_age`` is provided it will
        ensure the signature is not older than that time in seconds. In
        case the signature is outdated, :exc:`.SignatureExpired` is
        raised. All arguments are forwarded to the signer's
        :meth:`~TimestampSigner.unsign` method.
        """
        s = want_bytes(s)
        last_exception = None
        for signer in self.iter_unsigners(salt):
            try:
                base64d, timestamp = signer.unsign(s, max_age=max_age, return_timestamp=True)
                payload = self.load_payload(base64d)
                if return_timestamp:
                    return (payload, timestamp)
                return payload
            except SignatureExpired:
                raise
            except BadSignature as err:
                last_exception = err
        raise t.cast(BadSignature, last_exception)

    def loads_unsafe(self, s: str | bytes, max_age: int | None=None, salt: str | bytes | None=None) -> tuple[bool, t.Any]:
        return self._loads_unsafe_impl(s, salt, load_kwargs={'max_age': max_age})