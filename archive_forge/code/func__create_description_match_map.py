from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
@classmethod
def _create_description_match_map(cls, result_columns: List[ResultColumnsEntry], loose_column_name_matching: bool=False) -> Dict[Union[str, object], Tuple[str, Tuple[Any, ...], TypeEngine[Any], int]]:
    """when matching cursor.description to a set of names that are present
        in a Compiled object, as is the case with TextualSelect, get all the
        names we expect might match those in cursor.description.
        """
    d: Dict[Union[str, object], Tuple[str, Tuple[Any, ...], TypeEngine[Any], int]] = {}
    for ridx, elem in enumerate(result_columns):
        key = elem[RM_RENDERED_NAME]
        if key in d:
            e_name, e_obj, e_type, e_ridx = d[key]
            d[key] = (e_name, e_obj + elem[RM_OBJECTS], e_type, ridx)
        else:
            d[key] = (elem[RM_NAME], elem[RM_OBJECTS], elem[RM_TYPE], ridx)
        if loose_column_name_matching:
            for r_key in elem[RM_OBJECTS]:
                d.setdefault(r_key, (elem[RM_NAME], elem[RM_OBJECTS], elem[RM_TYPE], ridx))
    return d