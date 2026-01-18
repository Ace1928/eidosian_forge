import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _parse_keyexprs(self, identifiers):
    """Unpack '"col"(2),"col" ASC'-ish strings into components."""
    return [(colname, int(length) if length else None, modifiers) for colname, length, modifiers in self._re_keyexprs.findall(identifiers)]