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
def _adapt_to_context(self, context: ExecutionContext) -> ResultMetaData:
    """When using a cached Compiled construct that has a _result_map,
        for a new statement that used the cached Compiled, we need to ensure
        the keymap has the Column objects from our new statement as keys.
        So here we rewrite keymap with new entries for the new columns
        as matched to those of the cached statement.

        """
    if not context.compiled or not context.compiled._result_columns:
        return self
    compiled_statement = context.compiled.statement
    invoked_statement = context.invoked_statement
    if TYPE_CHECKING:
        assert isinstance(invoked_statement, elements.ClauseElement)
    if compiled_statement is invoked_statement:
        return self
    assert invoked_statement is not None
    keymap_by_position = self._keymap_by_result_column_idx
    if keymap_by_position is None:
        keymap_by_position = self._keymap_by_result_column_idx = {metadata_entry[MD_RESULT_MAP_INDEX]: metadata_entry for metadata_entry in self._keymap.values()}
    assert not self._tuplefilter
    return self._make_new_metadata(keymap=compat.dict_union(self._keymap, {new: keymap_by_position[idx] for idx, new in enumerate(invoked_statement._all_selected_columns) if idx in keymap_by_position}), unpickled=self._unpickled, processors=self._processors, tuplefilter=None, translated_indexes=None, keys=self._keys, safe_for_cache=self._safe_for_cache, keymap_by_result_column_idx=self._keymap_by_result_column_idx)