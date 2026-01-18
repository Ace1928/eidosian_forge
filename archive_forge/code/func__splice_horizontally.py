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
def _splice_horizontally(self, other: CursorResultMetaData) -> CursorResultMetaData:
    assert not self._tuplefilter
    keymap = dict(self._keymap)
    offset = len(self._keys)
    keymap.update({key: (value[0] + offset if value[0] is not None and key not in keymap else None, value[1] + offset, *value[2:]) for key, value in other._keymap.items()})
    return self._make_new_metadata(unpickled=self._unpickled, processors=self._processors + other._processors, tuplefilter=None, translated_indexes=None, keys=self._keys + other._keys, keymap=keymap, safe_for_cache=self._safe_for_cache, keymap_by_result_column_idx={metadata_entry[MD_RESULT_MAP_INDEX]: metadata_entry for metadata_entry in keymap.values()})