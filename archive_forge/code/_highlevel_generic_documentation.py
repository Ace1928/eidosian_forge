from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar
import attrs
import trio
from trio._util import final
from .abc import AsyncResource, HalfCloseableStream, ReceiveStream, SendStream
Calls ``aclose`` on both underlying streams.