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
def _classImplements_ordered(spec, before=(), after=()):
    before = [x for x in before if not spec.isOrExtends(x) or (x is Interface and (not spec.declared))]
    after = [x for x in after if not spec.isOrExtends(x) or (x is Interface and (not spec.declared))]
    new_declared = []
    seen = set()
    for l in (before, spec.declared, after):
        for b in l:
            if b not in seen:
                new_declared.append(b)
                seen.add(b)
    spec.declared = tuple(new_declared)
    bases = new_declared
    if spec.inherit is not None:
        for c in spec.inherit.__bases__:
            b = implementedBy(c)
            if b not in seen:
                seen.add(b)
                bases.append(b)
    spec.__bases__ = tuple(bases)