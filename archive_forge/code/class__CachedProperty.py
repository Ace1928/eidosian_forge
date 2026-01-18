from . import trace
class _CachedProperty:

    def __init__(self, attrname, fn):
        self.fn = fn
        self.attrname = attrname
        self.marker = object()

    def __get__(self, inst, cls=None):
        if inst is None:
            return self
        cachedresult = getattr(inst, self.attrname, self.marker)
        if cachedresult is self.marker:
            result = self.fn(inst)
            setattr(inst, self.attrname, result)
            return result
        else:
            return cachedresult