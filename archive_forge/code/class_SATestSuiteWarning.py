from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class SATestSuiteWarning(Warning):
    """warning for a condition detected during tests that is non-fatal

    Currently outside of SAWarning so that we can work around tools like
    Alembic doing the wrong thing with warnings.

    """