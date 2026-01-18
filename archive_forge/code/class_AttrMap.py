from reportlab.lib.validators import isAnything, DerivedValue
from reportlab.lib.utils import isSeq
from reportlab import rl_config
class AttrMap(dict):

    def __init__(self, BASE=None, UNWANTED=[], **kw):
        data = {}
        if BASE:
            if isinstance(BASE, AttrMap):
                data = BASE
            else:
                if not isSeq(BASE):
                    BASE = (BASE,)
                for B in BASE:
                    am = getattr(B, '_attrMap', self)
                    if am is not self:
                        if am:
                            data.update(am)
                    else:
                        raise ValueError('BASE=%s has wrong kind of value' % ascii(B))
        dict.__init__(self, data)
        self.remove(UNWANTED)
        self.update(kw)

    def remove(self, unwanted):
        for k in unwanted:
            try:
                del self[k]
            except KeyError:
                pass

    def clone(self, UNWANTED=[], **kw):
        c = AttrMap(BASE=self, UNWANTED=UNWANTED)
        c.update(kw)
        return c