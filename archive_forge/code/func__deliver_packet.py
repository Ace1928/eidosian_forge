from __future__ import annotations
import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
import attrs
import trio
from trio._util import NoPublicConstructor, final
def _deliver_packet(self, packet: UDPPacket) -> None:
    with contextlib.suppress(trio.BrokenResourceError):
        self._packet_sender.send_nowait(packet)