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
def _column_args_match(self, inspected_params: Params, meta_params: Params) -> bool:
    """We want to compare column parameters. However, we only want
        to compare parameters that are set. If they both have `collation`,
        we want to make sure they are the same. However, if only one
        specifies it, dont flag it for being less specific
        """
    if len(meta_params.tokens) == len(inspected_params.tokens) and meta_params.tokens != inspected_params.tokens:
        return False
    if len(meta_params.args) == len(inspected_params.args) and meta_params.args != inspected_params.args:
        return False
    insp = ' '.join(inspected_params.tokens).lower()
    meta = ' '.join(meta_params.tokens).lower()
    for reg in self.type_arg_extract:
        mi = re.search(reg, insp)
        mm = re.search(reg, meta)
        if mi and mm and (mi.group(1) != mm.group(1)):
            return False
    return True