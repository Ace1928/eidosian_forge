from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, overload
from . import validator
from .config import Extra
from .errors import ConfigError
from .main import BaseModel, create_model
from .typing import get_all_type_hints
from .utils import to_camel
class DecoratorBaseModel(BaseModel):

    @validator(self.v_args_name, check_fields=False, allow_reuse=True)
    def check_args(cls, v: Optional[List[Any]]) -> Optional[List[Any]]:
        if takes_args or v is None:
            return v
        raise TypeError(f'{pos_args} positional arguments expected but {pos_args + len(v)} given')

    @validator(self.v_kwargs_name, check_fields=False, allow_reuse=True)
    def check_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if takes_kwargs or v is None:
            return v
        plural = '' if len(v) == 1 else 's'
        keys = ', '.join(map(repr, v.keys()))
        raise TypeError(f'unexpected keyword argument{plural}: {keys}')

    @validator(V_POSITIONAL_ONLY_NAME, check_fields=False, allow_reuse=True)
    def check_positional_only(cls, v: Optional[List[str]]) -> None:
        if v is None:
            return
        plural = '' if len(v) == 1 else 's'
        keys = ', '.join(map(repr, v))
        raise TypeError(f'positional-only argument{plural} passed as keyword argument{plural}: {keys}')

    @validator(V_DUPLICATE_KWARGS, check_fields=False, allow_reuse=True)
    def check_duplicate_kwargs(cls, v: Optional[List[str]]) -> None:
        if v is None:
            return
        plural = '' if len(v) == 1 else 's'
        keys = ', '.join(map(repr, v))
        raise TypeError(f'multiple values for argument{plural}: {keys}')

    class Config(CustomConfig):
        extra = getattr(CustomConfig, 'extra', Extra.forbid)