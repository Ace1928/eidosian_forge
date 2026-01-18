import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _parse_values(s):
    """(INTERNAL) Split a line into a list of values"""
    if not _RE_NONTRIVIAL_DATA.search(s):
        return [None if s in ('?', '') else s for s in next(csv.reader([s]))]
    values, errors = zip(*_RE_DENSE_VALUES.findall(',' + s))
    if not any(errors):
        return [_unquote(v) for v in values]
    if _RE_SPARSE_LINE.match(s):
        try:
            return {int(k): _unquote(v) for k, v in _RE_SPARSE_KEY_VALUES.findall(s)}
        except ValueError:
            for match in _RE_SPARSE_KEY_VALUES.finditer(s):
                if not match.group(1):
                    raise BadLayout('Error parsing %r' % match.group())
            raise BadLayout('Unknown parsing error')
    else:
        for match in _RE_DENSE_VALUES.finditer(s):
            if match.group(2):
                raise BadLayout('Error parsing %r' % match.group())
        raise BadLayout('Unknown parsing error')