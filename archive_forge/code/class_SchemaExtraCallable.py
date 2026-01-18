import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, ForwardRef, Optional, Tuple, Type, Union
from typing_extensions import Literal, Protocol
from .typing import AnyArgTCallable, AnyCallable
from .utils import GetterDict
from .version import compiled
class SchemaExtraCallable(Protocol):

    @overload
    def __call__(self, schema: Dict[str, Any]) -> None:
        pass

    @overload
    def __call__(self, schema: Dict[str, Any], model_class: Type[BaseModel]) -> None:
        pass