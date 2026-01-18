from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from fastapi._compat import (
from fastapi.logger import logger
from pydantic import AnyUrl, BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict
from typing_extensions import deprecated as typing_deprecated
class Contact(BaseModel):
    name: Optional[str] = None
    url: Optional[AnyUrl] = None
    email: Optional[EmailStr] = None
    if PYDANTIC_V2:
        model_config = {'extra': 'allow'}
    else:

        class Config:
            extra = 'allow'