import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar, Union
from .parse import Protocol, load_file, load_str_bytes
from .types import StrBytes
from .typing import display_as_type
def _generate_parsing_type_name(type_: Any) -> str:
    return f'ParsingModel[{display_as_type(type_)}]'