from logging import Logger
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from uuid import uuid4
from triad import to_uuid
from fugue._utils.registry import fugue_plugin
from fugue._utils.misc import import_fsql_dependency
class TempTableName:
    """Generating a temporary, random and globaly unique table name"""

    def __init__(self):
        self.key = '_' + str(uuid4())[:5]

    def __repr__(self) -> str:
        return _TEMP_TABLE_EXPR_PREFIX + self.key + _TEMP_TABLE_EXPR_SUFFIX