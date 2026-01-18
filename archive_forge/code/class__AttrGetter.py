from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
class _AttrGetter:
    __slots__ = ('attr_name', 'getter')

    def __init__(self, attr_name: str):
        self.attr_name = attr_name
        self.getter = operator.attrgetter(attr_name)

    def __call__(self, mapped_object: Any) -> Any:
        obj = self.getter(mapped_object)
        if obj is None:
            state = base.instance_state(mapped_object)
            mp = state.mapper
            if self.attr_name in mp.attrs:
                dict_ = state.dict
                obj = dict_.get(self.attr_name, base.NO_VALUE)
                if obj is None:
                    return _UNMAPPED_AMBIGUOUS_NONE
            else:
                return _UNMAPPED_AMBIGUOUS_NONE
        return obj

    def __reduce__(self) -> Tuple[Type[_AttrGetter], Tuple[str]]:
        return (_AttrGetter, (self.attr_name,))