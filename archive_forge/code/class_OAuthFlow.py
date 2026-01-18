from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from fastapi._compat import (
from fastapi.logger import logger
from pydantic import AnyUrl, BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict
from typing_extensions import deprecated as typing_deprecated
class OAuthFlow(BaseModel):
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str] = {}
    if PYDANTIC_V2:
        model_config = {'extra': 'allow'}
    else:

        class Config:
            extra = 'allow'