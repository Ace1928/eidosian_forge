from __future__ import annotations
import asyncio
import urllib.parse
from uvicorn._types import WWWScope
class ClientDisconnected(IOError):
    ...