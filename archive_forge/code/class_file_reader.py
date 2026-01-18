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
class file_reader(Generic[T]):

    def __init__(self, fo_or_decoder, reader_schema=None, options={}):
        if isinstance(fo_or_decoder, AvroJSONDecoder):
            self.decoder = fo_or_decoder
        else:
            self.decoder = BinaryDecoder(fo_or_decoder)
        self._named_schemas = _default_named_schemas()
        if reader_schema:
            self.reader_schema = parse_schema(reader_schema, self._named_schemas['reader'], _write_hint=False)
        else:
            self.reader_schema = None
        self.options = options
        self._elems = None

    def _read_header(self):
        try:
            self._header = read_data(self.decoder, HEADER_SCHEMA, self._named_schemas, None, self.options)
        except EOFError:
            raise ValueError('cannot read header - is it an avro file?')
        self.metadata = {k: v.decode() for k, v in self._header['meta'].items()}
        self._schema = json.loads(self.metadata['avro.schema'])
        self.codec = self.metadata.get('avro.codec', 'null')
        if self.reader_schema is not None:
            ignore_default_error = True
        else:
            ignore_default_error = False
        self.writer_schema = parse_schema(self._schema, self._named_schemas['writer'], _write_hint=False, _force=True, _ignore_default_error=ignore_default_error)

    @property
    def schema(self):
        import warnings
        warnings.warn("The 'schema' attribute is deprecated. Please use 'writer_schema'", DeprecationWarning)
        return self._schema

    def __iter__(self) -> Iterator[T]:
        if not self._elems:
            raise NotImplementedError
        return self._elems

    def __next__(self) -> T:
        return next(self._elems)