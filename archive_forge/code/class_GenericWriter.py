from abc import ABC, abstractmethod
import json
from io import BytesIO
from os import urandom, SEEK_SET
import bz2
import lzma
import zlib
from typing import Union, IO, Iterable, Any, Optional, Dict
from warnings import warn
from .const import NAMED_TYPES
from .io.binary_encoder import BinaryEncoder
from .io.json_encoder import AvroJSONEncoder
from .validation import _validate
from .read import HEADER_SCHEMA, SYNC_SIZE, MAGIC, reader
from .logical_writers import LOGICAL_WRITERS
from .schema import extract_record_type, extract_logical_type, parse_schema
from ._write_common import _is_appendable
from .types import Schema, NamedSchemas
class GenericWriter(ABC):

    def __init__(self, schema, metadata=None, validator=None, options={}):
        self._named_schemas = {}
        self.validate_fn = _validate if validator else None
        self.metadata = metadata or {}
        self.options = options
        if schema is not None:
            self.schema = parse_schema(schema, self._named_schemas)
        if isinstance(schema, dict):
            schema = {key: value for key, value in schema.items() if key not in ('__fastavro_parsed', '__named_schemas')}
        elif isinstance(schema, list):
            schemas = []
            for s in schema:
                if isinstance(s, dict):
                    schemas.append({key: value for key, value in s.items() if key not in ('__fastavro_parsed', '__named_schemas')})
                else:
                    schemas.append(s)
            schema = schemas
        self.metadata['avro.schema'] = json.dumps(schema)

    @abstractmethod
    def write(self, record):
        pass

    @abstractmethod
    def flush(self):
        pass