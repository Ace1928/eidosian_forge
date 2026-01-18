from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def _send_packet(self, command: int, data: bytes) -> None:
    """
        low-level packet sending.
        Following the protocol requires waiting for ack packet between
        sending each packet to the device.
        """
    buf = bytearray([command, len(data)])
    buf.extend(data)
    buf.extend(self.get_crc(buf))
    self._device.write(buf)