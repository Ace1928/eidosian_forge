import enum
import struct
import typing
def _pack_value(addr_type: typing.Optional['AddressType'], b: typing.Optional[bytes]) -> bytes:
    """Packs an type/data entry into the byte structure required."""
    if not b:
        b = b''
    return (struct.pack('<I', addr_type) if addr_type is not None else b'') + struct.pack('<I', len(b)) + b