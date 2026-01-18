from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from fastapi._compat import (
from fastapi.logger import logger
from pydantic import AnyUrl, BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict
from typing_extensions import deprecated as typing_deprecated
@classmethod
def __get_pydantic_core_schema__(cls, source: Type[Any], handler: Callable[[Any], CoreSchema]) -> CoreSchema:
    return with_info_plain_validator_function(cls._validate)