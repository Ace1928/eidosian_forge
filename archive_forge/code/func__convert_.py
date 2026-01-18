import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _convert_(cls, name, module, filter, source=None, *, boundary=None, as_global=False):
    """
        Create a new Enum subclass that replaces a collection of global constants
        """
    module_globals = sys.modules[module].__dict__
    if source:
        source = source.__dict__
    else:
        source = module_globals
    members = [(name, value) for name, value in source.items() if filter(name)]
    try:
        members.sort(key=lambda t: (t[1], t[0]))
    except TypeError:
        members.sort(key=lambda t: t[0])
    body = {t[0]: t[1] for t in members}
    body['__module__'] = module
    tmp_cls = type(name, (object,), body)
    cls = _simple_enum(etype=cls, boundary=boundary or KEEP)(tmp_cls)
    if as_global:
        global_enum(cls)
    else:
        sys.modules[cls.__module__].__dict__.update(cls.__members__)
    module_globals[name] = cls
    return cls