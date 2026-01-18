import struct
from typing import cast, Dict, List, Tuple, Union
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
def _decode_boolean(self, size: int, offset: int) -> Tuple[bool, int]:
    return (size != 0, offset)