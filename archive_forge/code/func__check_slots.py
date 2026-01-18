from abc import ABC
from ..coretypes import (
from ..dispatch import dispatch
@assert_dshape_equal.register(Slotted, Slotted)
def _check_slots(a, b, path=None, **kwargs):
    if type(a) != type(b):
        return _base_case(a, b, path=path, **kwargs)
    assert a.__slots__ == b.__slots__, 'slots mismatch: %r != %r\n%s' % (a.__slots__, b.__slots__, _fmt_path(path))
    if path is None:
        path = ()
    for slot in a.__slots__:
        assert getattr(a, slot) == getattr(b, slot), '%s %ss do not match: %r != %r\n%s' % (type(a).__name__.lower(), slot, getattr(a, slot), getattr(b, slot), _fmt_path(path + ('.' + slot,)))