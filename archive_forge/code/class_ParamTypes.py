import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing_extensions import Annotated, deprecated
from ._compat import PYDANTIC_V2, Undefined
class ParamTypes(Enum):
    query = 'query'
    header = 'header'
    path = 'path'
    cookie = 'cookie'