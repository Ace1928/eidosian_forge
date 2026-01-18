from __future__ import print_function, absolute_import, division, unicode_literals
from .compat import no_limit_int  # NOQA
from ruamel.yaml.anchor import Anchor
class ScalarInt(no_limit_int):

    def __new__(cls, *args, **kw):
        width = kw.pop('width', None)
        underscore = kw.pop('underscore', None)
        anchor = kw.pop('anchor', None)
        v = no_limit_int.__new__(cls, *args, **kw)
        v._width = width
        v._underscore = underscore
        if anchor is not None:
            v.yaml_set_anchor(anchor, always_dump=True)
        return v

    def __iadd__(self, a):
        x = type(self)(self + a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __ifloordiv__(self, a):
        x = type(self)(self // a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __imul__(self, a):
        x = type(self)(self * a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __ipow__(self, a):
        x = type(self)(self ** a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __isub__(self, a):
        x = type(self)(self - a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    @property
    def anchor(self):
        if not hasattr(self, Anchor.attrib):
            setattr(self, Anchor.attrib, Anchor())
        return getattr(self, Anchor.attrib)

    def yaml_anchor(self, any=False):
        if not hasattr(self, Anchor.attrib):
            return None
        if any or self.anchor.always_dump:
            return self.anchor
        return None

    def yaml_set_anchor(self, value, always_dump=False):
        self.anchor.value = value
        self.anchor.always_dump = always_dump