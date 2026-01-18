import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _pr_compile(regex, cleanup=None):
    """Prepare a 2-tuple of compiled regex and callable."""
    return (_re_compile(regex), cleanup)