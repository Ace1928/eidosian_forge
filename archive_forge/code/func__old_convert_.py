import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _old_convert_(etype, name, module, filter, source=None, *, boundary=None):
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
    cls = etype(name, members, module=module, boundary=boundary or KEEP)
    return cls