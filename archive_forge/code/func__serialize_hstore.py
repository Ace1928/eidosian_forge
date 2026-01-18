import re
from .array import ARRAY
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import GETITEM
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from ... import types as sqltypes
from ...sql import functions as sqlfunc
def _serialize_hstore(val):
    """Serialize a dictionary into an hstore literal.  Keys and values must
    both be strings (except None for values).

    """

    def esc(s, position):
        if position == 'value' and s is None:
            return 'NULL'
        elif isinstance(s, str):
            return '"%s"' % s.replace('\\', '\\\\').replace('"', '\\"')
        else:
            raise ValueError('%r in %s position is not a string.' % (s, position))
    return ', '.join(('%s=>%s' % (esc(k, 'key'), esc(v, 'value')) for k, v in val.items()))