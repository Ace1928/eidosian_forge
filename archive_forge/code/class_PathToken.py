from __future__ import annotations
from functools import reduce
from itertools import chain
import logging
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import base as orm_base
from ._typing import insp_is_mapper_property
from .. import exc
from .. import util
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
class PathToken(orm_base.InspectionAttr, HasCacheKey, str):
    """cacheable string token"""
    _intern: Dict[str, PathToken] = {}

    def _gen_cache_key(self, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Tuple[Any, ...]:
        return (str(self),)

    @property
    def _path_for_compare(self) -> Optional[_PathRepresentation]:
        return None

    @classmethod
    def intern(cls, strvalue: str) -> PathToken:
        if strvalue in cls._intern:
            return cls._intern[strvalue]
        else:
            cls._intern[strvalue] = result = PathToken(strvalue)
            return result