import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
class SingleHost:
    """Single_Host_Data structure for NTLM TargetInfo entry.

    `Single_Host_Data`_ structure allows a client to send machine-specific information within an authentication
    exchange to services on the same machine. If the server and client platforms are different or if they are on
    different hosts, then the information MUST be ignores.

    Args:
        size: A 32-bit unsigned int that defines size of the structure.
        z4: A 32-bit integer value, currently set to 0.
        custom_data: An 8-byte platform-specific blob containing info only relevant when the client and server are on
            the same host.
        machine_id: A 32-byte random number created at computer startup to identify the calling machine.
        _b_data: Create a SingleHost object from the raw data byte string.

    Attributes:
        size: See args.
        z4: See args.
        custom_data: See args.
        machine_id: See args.

    .. _Single_Host_Data:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/f221c061-cc40-4471-95da-d2ff71c85c5b
    """

    def __init__(self, size: int=0, z4: int=0, custom_data: typing.Optional[bytes]=None, machine_id: typing.Optional[bytes]=None, _b_data: typing.Optional[bytes]=None) -> None:
        if _b_data:
            if len(_b_data) != 48:
                raise ValueError('SingleHost bytes must have a length of 48')
            self._data = memoryview(_b_data)
        else:
            self._data = memoryview(bytearray(48))
            self.size = size
            self.z4 = z4
            self.custom_data = custom_data or b'\x00' * 8
            self.machine_id = machine_id or b'\x00' * 32

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (bytes, SingleHost)):
            return False
        if isinstance(other, SingleHost):
            other = other.pack()
        return self.pack() == other

    @property
    def size(self) -> int:
        return struct.unpack('<I', self._data[:4].tobytes())[0]

    @size.setter
    def size(self, value: int) -> None:
        self._data[:4] = struct.pack('<I', value)

    @property
    def z4(self) -> int:
        return struct.unpack('<I', self._data[4:8].tobytes())[0]

    @z4.setter
    def z4(self, value: int) -> None:
        self._data[4:8] = struct.pack('<I', value)

    @property
    def custom_data(self) -> bytes:
        return self._data[8:16].tobytes()

    @custom_data.setter
    def custom_data(self, value: bytes) -> None:
        if len(value) != 8:
            raise ValueError('custom_data length must be 8 bytes long')
        self._data[8:16] = value

    @property
    def machine_id(self) -> bytes:
        return self._data[16:48].tobytes()

    @machine_id.setter
    def machine_id(self, value: bytes) -> None:
        if len(value) != 32:
            raise ValueError('machine_id length must be 32 bytes long')
        self._data[16:48] = value

    def pack(self) -> bytes:
        """Packs the structure to bytes."""
        return self._data.tobytes()

    @staticmethod
    def unpack(b_data: bytes) -> 'SingleHost':
        """Creates a SignleHost object from raw bytes."""
        return SingleHost(_b_data=b_data)