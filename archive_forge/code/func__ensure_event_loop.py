import asyncio
from contextlib import contextmanager
from typing import Iterator
import httpx
from qcs_api_client.client import QCSClientConfiguration, build_sync_client
def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError as ex:
        if len(ex.args) > 0 and 'There is no current event loop in thread' in ex.args[0]:
            asyncio.set_event_loop(asyncio.new_event_loop())
        else:
            raise