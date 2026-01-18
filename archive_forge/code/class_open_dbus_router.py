import asyncio
import contextlib
from itertools import count
from typing import Optional
from jeepney.auth import Authenticator, BEGIN
from jeepney.bus import get_bus
from jeepney import Message, MessageType, Parser
from jeepney.wrappers import ProxyBase, unwrap_msg
from jeepney.bus_messages import message_bus
from .common import (
class open_dbus_router:
    """Open a D-Bus 'router' to send and receive messages

    Use as an async context manager::

        async with open_dbus_router() as router:
            ...
    """
    conn = None
    req_ctx = None

    def __init__(self, bus='SESSION'):
        self.bus = bus

    async def __aenter__(self):
        self.conn = await open_dbus_connection(self.bus)
        self.req_ctx = DBusRouter(self.conn)
        return await self.req_ctx.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.req_ctx.__aexit__(exc_type, exc_val, exc_tb)
        await self.conn.close()