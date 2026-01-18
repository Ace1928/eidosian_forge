import bz2
import json
import lzma
import zlib
from datetime import datetime, timezone
from decimal import Context
from io import BytesIO
from struct import error as StructError
from typing import IO, Union, Optional, Generic, TypeVar, Iterator, Dict
from warnings import warn
from .io.binary_decoder import BinaryDecoder
from .io.json_decoder import AvroJSONDecoder
from .logical_readers import LOGICAL_READERS
from .schema import (
from .types import Schema, AvroMessage, NamedSchemas
from ._read_common import (
from .const import NAMED_TYPES, AVRO_TYPES
def _iter_avro_blocks(decoder, header, codec, writer_schema, named_schemas, reader_schema, options):
    """Return iterator over avro blocks."""
    sync_marker = header['sync']
    read_block = BLOCK_READERS.get(codec)
    if not read_block:
        raise ValueError(f'Unrecognized codec: {codec}')
    while True:
        offset = decoder.fo.tell()
        try:
            num_block_records = decoder.read_long()
        except EOFError:
            return
        block_bytes = read_block(decoder)
        skip_sync(decoder.fo, sync_marker)
        size = decoder.fo.tell() - offset
        yield Block(block_bytes, num_block_records, codec, reader_schema, writer_schema, named_schemas, offset, size, options)