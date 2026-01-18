import struct
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union
import numpy as np
from ray.data.block import Block
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _write_record(file: 'pyarrow.NativeFile', example: 'tf.train.Example') -> None:
    record = example.SerializeToString()
    length = len(record)
    length_bytes = struct.pack('<Q', length)
    file.write(length_bytes)
    file.write(_masked_crc(length_bytes))
    file.write(record)
    file.write(_masked_crc(record))