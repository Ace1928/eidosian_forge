import sys
import weakref
from types import FunctionType
from types import MethodType
from types import ModuleType
from zope.interface._compat import _use_c_impl
from zope.interface.interface import Interface
from zope.interface.interface import InterfaceClass
from zope.interface.interface import NameAndModuleComparisonMixin
from zope.interface.interface import Specification
from zope.interface.interface import SpecificationBase
def _implementedBy_super(sup):
    implemented_by_self = implementedBy(sup.__self_class__)
    cache = implemented_by_self._super_cache
    if cache is None:
        cache = implemented_by_self._super_cache = weakref.WeakKeyDictionary()
    key = sup.__thisclass__
    try:
        return cache[key]
    except KeyError:
        pass
    next_cls = _next_super_class(sup)
    implemented_by_next = implementedBy(next_cls)
    mro = sup.__self_class__.__mro__
    ix_next_cls = mro.index(next_cls)
    classes_to_keep = mro[ix_next_cls:]
    new_bases = [implementedBy(c) for c in classes_to_keep]
    new = Implements.named(implemented_by_self.__name__ + ':' + implemented_by_next.__name__, *new_bases)
    new.inherit = implemented_by_next.inherit
    new.declared = implemented_by_next.declared
    cache[key] = new
    return new