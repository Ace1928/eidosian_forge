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
def _normalizeargs(sequence, output=None):
    """Normalize declaration arguments

    Normalization arguments might contain Declarions, tuples, or single
    interfaces.

    Anything but individual interfaces or implements specs will be expanded.
    """
    if output is None:
        output = []
    cls = sequence.__class__
    if InterfaceClass in cls.__mro__ or Implements in cls.__mro__:
        output.append(sequence)
    else:
        for v in sequence:
            _normalizeargs(v, output)
    return output