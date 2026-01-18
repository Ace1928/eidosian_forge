from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from fastapi._compat import (
from fastapi.logger import logger
from pydantic import AnyUrl, BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict
from typing_extensions import deprecated as typing_deprecated
class Components(BaseModel):
    schemas: Optional[Dict[str, Union[Schema, Reference]]] = None
    responses: Optional[Dict[str, Union[Response, Reference]]] = None
    parameters: Optional[Dict[str, Union[Parameter, Reference]]] = None
    examples: Optional[Dict[str, Union[Example, Reference]]] = None
    requestBodies: Optional[Dict[str, Union[RequestBody, Reference]]] = None
    headers: Optional[Dict[str, Union[Header, Reference]]] = None
    securitySchemes: Optional[Dict[str, Union[SecurityScheme, Reference]]] = None
    links: Optional[Dict[str, Union[Link, Reference]]] = None
    callbacks: Optional[Dict[str, Union[Dict[str, PathItem], Reference, Any]]] = None
    pathItems: Optional[Dict[str, Union[PathItem, Reference]]] = None
    if PYDANTIC_V2:
        model_config = {'extra': 'allow'}
    else:

        class Config:
            extra = 'allow'