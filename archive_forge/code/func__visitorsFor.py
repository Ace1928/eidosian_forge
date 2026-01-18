import enum
@classmethod
def _visitorsFor(celf, thing, _default={}):
    typ = type(thing)
    for celf in celf.mro():
        _visitors = getattr(celf, '_visitors', None)
        if _visitors is None:
            break
        m = celf._visitors.get(typ, None)
        if m is not None:
            return m
    return _default