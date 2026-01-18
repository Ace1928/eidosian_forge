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
def _elems():
    while not self.decoder.done:
        yield read_data(self.decoder, self.writer_schema, self._named_schemas, self.reader_schema, self.options)
        self.decoder.drain()