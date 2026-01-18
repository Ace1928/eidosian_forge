from __future__ import annotations
import codecs
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any
from ..abc import (
@dataclass(eq=False)
class TextReceiveStream(ObjectReceiveStream[str]):
    """
    Stream wrapper that decodes bytes to strings using the given encoding.

    Decoding is done using :class:`~codecs.IncrementalDecoder` which returns any
    completely received unicode characters as soon as they come in.

    :param transport_stream: any bytes-based receive stream
    :param encoding: character encoding to use for decoding bytes to strings (defaults
        to ``utf-8``)
    :param errors: handling scheme for decoding errors (defaults to ``strict``; see the
        `codecs module documentation`_ for a comprehensive list of options)

    .. _codecs module documentation:
        https://docs.python.org/3/library/codecs.html#codec-objects
    """
    transport_stream: AnyByteReceiveStream
    encoding: InitVar[str] = 'utf-8'
    errors: InitVar[str] = 'strict'
    _decoder: codecs.IncrementalDecoder = field(init=False)

    def __post_init__(self, encoding: str, errors: str) -> None:
        decoder_class = codecs.getincrementaldecoder(encoding)
        self._decoder = decoder_class(errors=errors)

    async def receive(self) -> str:
        while True:
            chunk = await self.transport_stream.receive()
            decoded = self._decoder.decode(chunk)
            if decoded:
                return decoded

    async def aclose(self) -> None:
        await self.transport_stream.aclose()
        self._decoder.reset()

    @property
    def extra_attributes(self) -> Mapping[Any, Callable[[], Any]]:
        return self.transport_stream.extra_attributes