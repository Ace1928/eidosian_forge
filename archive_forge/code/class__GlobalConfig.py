import functools
from enum import Enum
from typing import Callable, Dict, Optional, TypeVar, Union
from marshmallow.fields import Field as MarshmallowField  # type: ignore
from dataclasses_json.stringcase import (camelcase, pascalcase, snakecase,
from dataclasses_json.undefined import Undefined, UndefinedParameterError
class _GlobalConfig:

    def __init__(self):
        self.encoders: Dict[Union[type, Optional[type]], Callable] = {}
        self.decoders: Dict[Union[type, Optional[type]], Callable] = {}
        self.mm_fields: Dict[Union[type, Optional[type]], MarshmallowField] = {}