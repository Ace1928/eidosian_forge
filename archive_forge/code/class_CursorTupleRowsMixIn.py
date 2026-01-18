import re
from ._exceptions import ProgrammingError
class CursorTupleRowsMixIn:
    """This is a MixIn class that causes all rows to be returned as tuples,
    which is the standard form required by DB API."""
    _fetch_type = 0