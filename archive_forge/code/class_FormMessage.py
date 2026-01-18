from __future__ import annotations
import typing
from dataclasses import dataclass, field
from enum import Enum
from tempfile import SpooledTemporaryFile
from urllib.parse import unquote_plus
from starlette.datastructures import FormData, Headers, UploadFile
class FormMessage(Enum):
    FIELD_START = 1
    FIELD_NAME = 2
    FIELD_DATA = 3
    FIELD_END = 4
    END = 5