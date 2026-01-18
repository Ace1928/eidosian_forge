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
def _parse_hstore(hstore_str):
    """Parse an hstore from its literal string representation.

    Attempts to approximate PG's hstore input parsing rules as closely as
    possible. Although currently this is not strictly necessary, since the
    current implementation of hstore's output syntax is stricter than what it
    accepts as input, the documentation makes no guarantees that will always
    be the case.



    """
    result = {}
    pos = 0
    pair_match = HSTORE_PAIR_RE.match(hstore_str)
    while pair_match is not None:
        key = pair_match.group('key').replace('\\"', '"').replace('\\\\', '\\')
        if pair_match.group('value_null'):
            value = None
        else:
            value = pair_match.group('value').replace('\\"', '"').replace('\\\\', '\\')
        result[key] = value
        pos += pair_match.end()
        delim_match = HSTORE_DELIMITER_RE.match(hstore_str[pos:])
        if delim_match is not None:
            pos += delim_match.end()
        pair_match = HSTORE_PAIR_RE.match(hstore_str[pos:])
    if pos != len(hstore_str):
        raise ValueError(_parse_error(hstore_str, pos))
    return result