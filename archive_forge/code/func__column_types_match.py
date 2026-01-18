from __future__ import annotations
import logging
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import cast
from sqlalchemy import schema
from sqlalchemy import text
from . import _autogen
from . import base
from ._autogen import _constraint_sig as _constraint_sig
from ._autogen import ComparisonResult as ComparisonResult
from .. import util
from ..util import sqla_compat
def _column_types_match(self, inspector_params: Params, metadata_params: Params) -> bool:
    if inspector_params.token0 == metadata_params.token0:
        return True
    synonyms = [{t.lower() for t in batch} for batch in self.type_synonyms]
    inspector_all_terms = ' '.join([inspector_params.token0] + inspector_params.tokens)
    metadata_all_terms = ' '.join([metadata_params.token0] + metadata_params.tokens)
    for batch in synonyms:
        if {inspector_all_terms, metadata_all_terms}.issubset(batch) or {inspector_params.token0, metadata_params.token0}.issubset(batch):
            return True
    return False