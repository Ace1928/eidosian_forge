from __future__ import print_function, absolute_import, division, unicode_literals
import sys
from .compat import no_limit_int  # NOQA
from ruamel.yaml.anchor import Anchor
class ScalarFloat(float):

    def __new__(cls, *args, **kw):
        width = kw.pop('width', None)
        prec = kw.pop('prec', None)
        m_sign = kw.pop('m_sign', None)
        m_lead0 = kw.pop('m_lead0', 0)
        exp = kw.pop('exp', None)
        e_width = kw.pop('e_width', None)
        e_sign = kw.pop('e_sign', None)
        underscore = kw.pop('underscore', None)
        anchor = kw.pop('anchor', None)
        v = float.__new__(cls, *args, **kw)
        v._width = width
        v._prec = prec
        v._m_sign = m_sign
        v._m_lead0 = m_lead0
        v._exp = exp
        v._e_width = e_width
        v._e_sign = e_sign
        v._underscore = underscore
        if anchor is not None:
            v.yaml_set_anchor(anchor, always_dump=True)
        return v

    def __iadd__(self, a):
        return float(self) + a
        x = type(self)(self + a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __ifloordiv__(self, a):
        return float(self) // a
        x = type(self)(self // a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __imul__(self, a):
        return float(self) * a
        x = type(self)(self * a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        x._prec = self._prec
        return x

    def __ipow__(self, a):
        return float(self) ** a
        x = type(self)(self ** a)
        x._width = self._width
        x._underscore = self._underscore[:] if self._underscore is not None else None
        return x

    def __isub__(self, a):
        return float(self) - a
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

    def dump(self, out=sys.stdout):
        out.write('ScalarFloat({}| w:{}, p:{}, s:{}, lz:{}, _:{}|{}, w:{}, s:{})\n'.format(self, self._width, self._prec, self._m_sign, self._m_lead0, self._underscore, self._exp, self._e_width, self._e_sign))