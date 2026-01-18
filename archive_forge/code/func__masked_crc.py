import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _masked_crc(data: bytes) -> bytes:
    """CRC checksum."""
    import crc32c
    mask = 2726488792
    crc = crc32c.crc32(data)
    masked = (crc >> 15 | crc << 17) + mask
    masked = np.uint32(masked & np.iinfo(np.uint32).max)
    masked_bytes = struct.pack('<I', masked)
    return masked_bytes