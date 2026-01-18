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
def _missing_codec_lib(codec, *libraries):

    def missing(encoder, block_bytes, compression_level):
        raise ValueError(f'{codec} codec is supported but you need to install one of the ' + f'following libraries: {libraries}')
    return missing