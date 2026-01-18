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
class implementer_only:
    """Declare the only interfaces implemented by instances of a class

      This function is called as a class decorator.

      The arguments are one or more interfaces or interface
      specifications (`~zope.interface.interfaces.IDeclaration` objects).

      Previous declarations including declarations for base classes
      are overridden.

      This function is provided for convenience. It provides a more
      convenient way to call `classImplementsOnly`. For example::

        @implementer_only(I1)
        class C(object): pass

      is equivalent to calling::

        classImplementsOnly(I1)

      after the class has been created.
      """

    def __init__(self, *interfaces):
        self.interfaces = interfaces

    def __call__(self, ob):
        if isinstance(ob, (FunctionType, MethodType)):
            raise ValueError('The implementer_only decorator is not supported for methods or functions.')
        classImplementsOnly(ob, *self.interfaces)
        return ob