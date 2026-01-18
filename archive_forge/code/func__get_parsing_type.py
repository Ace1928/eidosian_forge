import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar, Union
from .parse import Protocol, load_file, load_str_bytes
from .types import StrBytes
from .typing import display_as_type
@lru_cache(maxsize=2048)
def _get_parsing_type(type_: Any, *, type_name: Optional[NameFactory]=None) -> Any:
    from .main import create_model
    if type_name is None:
        type_name = _generate_parsing_type_name
    if not isinstance(type_name, str):
        type_name = type_name(type_)
    return create_model(type_name, __root__=(type_, ...))