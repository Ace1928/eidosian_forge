from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
class CFLCDScreen(LCDScreen, abc.ABC):
    """
    Common methods for Crystal Fonts LCD displays
    """
    KEYS: typing.ClassVar[list[str | None]] = [None, 'up_press', 'down_press', 'left_press', 'right_press', 'enter_press', 'exit_press', 'up_release', 'down_release', 'left_release', 'right_release', 'enter_release', 'exit_release', 'ul_press', 'ur_press', 'll_press', 'lr_press', 'ul_release', 'ur_release', 'll_release', 'lr_release']
    CMD_PING = 0
    CMD_VERSION = 1
    CMD_CLEAR = 6
    CMD_CGRAM = 9
    CMD_CURSOR_POSITION = 11
    CMD_CURSOR_STYLE = 12
    CMD_LCD_CONTRAST = 13
    CMD_BACKLIGHT = 14
    CMD_LCD_DATA = 31
    CMD_GPO = 34
    CMD_KEY_ACTIVITY = 128
    CMD_ACK = 64
    CURSOR_NONE = 0
    CURSOR_BLINKING_BLOCK = 1
    CURSOR_UNDERSCORE = 2
    CURSOR_BLINKING_BLOCK_UNDERSCORE = 3
    CURSOR_INVERTING_BLINKING_BLOCK = 4
    MAX_PACKET_DATA_LENGTH = 22
    colors = 1
    has_underline = False

    def __init__(self, device_path: str, baud: int) -> None:
        """
        device_path -- eg. '/dev/ttyUSB0'
        baud -- baud rate
        """
        super().__init__()
        self.device_path = device_path
        from serial import Serial
        self._device = Serial(device_path, baud, timeout=0)
        self._unprocessed = bytearray()

    @classmethod
    def get_crc(cls, buf: Iterable[int]) -> bytes:
        new_crc = 15933696
        for byte in buf:
            for bit_count in range(8):
                new_crc >>= 1
                if byte & 1 << bit_count:
                    new_crc |= 8388608
                if new_crc & 128:
                    new_crc ^= 8652800
        for _bit_count in range(16):
            new_crc >>= 1
            if new_crc & 128:
                new_crc ^= 8652800
        return (~new_crc >> 8 & 65535).to_bytes(2, 'little')

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

    def _read_packet(self) -> tuple[int, bytearray] | None:
        """
        low-level packet reading.
        returns (command/report code, data) or None

        This method stored data read and tries to resync when bad data
        is received.
        """
        self._unprocessed += self._device.read()
        while True:
            try:
                command, data, unprocessed = self._parse_data(self._unprocessed)
                self._unprocessed = unprocessed
            except self.MoreDataRequired:
                return None
            except self.InvalidPacket:
                self._unprocessed = self._unprocessed[1:]
            else:
                return (command, data)

    class InvalidPacket(Exception):
        pass

    class MoreDataRequired(Exception):
        pass

    @classmethod
    def _parse_data(cls, data: bytearray) -> tuple[int, bytearray, bytearray]:
        """
        Try to read a packet from the start of data, returning
        (command/report code, packet_data, remaining_data)
        or raising InvalidPacket or MoreDataRequired
        """
        if len(data) < 2:
            raise cls.MoreDataRequired
        command: int = data[0]
        packet_len: int = data[1]
        if packet_len > cls.MAX_PACKET_DATA_LENGTH:
            raise cls.InvalidPacket('length value too large')
        if len(data) < packet_len + 4:
            raise cls.MoreDataRequired
        data_end = 2 + packet_len
        crc = cls.get_crc(data[:data_end])
        pcrc = data[data_end:data_end + 2]
        if crc != pcrc:
            raise cls.InvalidPacket("CRC doesn't match")
        return (command, data[2:data_end], data[data_end + 2:])