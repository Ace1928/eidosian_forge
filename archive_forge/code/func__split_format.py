import collections
import enum
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables.otConverters import (
from fontTools.misc.roundTools import otRound
def _split_format(cls, source):
    if _isNonStrSequence(source):
        assert len(source) > 0, f'{cls} needs at least format from {source}'
        fmt, remainder = (source[0], source[1:])
    elif isinstance(source, collections.abc.Mapping):
        assert 'Format' in source, f'{cls} needs at least Format from {source}'
        remainder = source.copy()
        fmt = remainder.pop('Format')
    else:
        raise ValueError(f'Not sure how to populate {cls} from {source}')
    assert isinstance(fmt, collections.abc.Hashable), f'{cls} Format is not hashable: {fmt!r}'
    assert fmt in cls.convertersByName, f'{cls} invalid Format: {fmt!r}'
    return (fmt, remainder)