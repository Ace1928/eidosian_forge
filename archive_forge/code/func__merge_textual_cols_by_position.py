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
def _merge_textual_cols_by_position(self, context, cursor_description, result_columns):
    num_ctx_cols = len(result_columns)
    if num_ctx_cols > len(cursor_description):
        util.warn('Number of columns in textual SQL (%d) is smaller than number of columns requested (%d)' % (num_ctx_cols, len(cursor_description)))
    seen = set()
    for idx, colname, untranslated, coltype in self._colnames_from_description(context, cursor_description):
        if idx < num_ctx_cols:
            ctx_rec = result_columns[idx]
            obj = ctx_rec[RM_OBJECTS]
            ridx = idx
            mapped_type = ctx_rec[RM_TYPE]
            if obj[0] in seen:
                raise exc.InvalidRequestError('Duplicate column expression requested in textual SQL: %r' % obj[0])
            seen.add(obj[0])
        else:
            mapped_type = sqltypes.NULLTYPE
            obj = None
            ridx = None
        yield (idx, ridx, colname, mapped_type, coltype, obj, untranslated)