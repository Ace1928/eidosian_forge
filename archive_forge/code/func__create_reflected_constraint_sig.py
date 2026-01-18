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
def _create_reflected_constraint_sig(self, constraint: _autogen._C, **opts: Any) -> _constraint_sig[_autogen._C]:
    return _constraint_sig.from_constraint(False, self, constraint, **opts)