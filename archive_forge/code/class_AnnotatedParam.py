import copy
import inspect
import re
from typing import (
from ..exceptions import InvalidOperationError
from ..utils.assertion import assert_or_throw
from ..utils.convert import get_full_type_path
from ..utils.entry_points import load_entry_point
from ..utils.hash import to_uuid
from .dict import IndexedOrderedDict
class AnnotatedParam:
    """An abstraction of annotated parameter"""

    def __init__(self, param: Optional[inspect.Parameter]):
        if param is not None:
            self.required = param.default == inspect.Parameter.empty
            self.default = param.default
        else:
            self.required, self.default = (True, None)
        self.annotation: Any = getattr(self.__class__, '_annotation')
        self.code: str = getattr(self.__class__, '_code')

    def __repr__(self) -> str:
        return str(self.annotation)