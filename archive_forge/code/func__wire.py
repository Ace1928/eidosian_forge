import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
def _wire():
    from zope.interface.declarations import classImplements
    from zope.interface.interfaces import IElement
    classImplements(Element, IElement)
    from zope.interface.interfaces import IAttribute
    classImplements(Attribute, IAttribute)
    from zope.interface.interfaces import IMethod
    classImplements(Method, IMethod)
    from zope.interface.interfaces import ISpecification
    classImplements(Specification, ISpecification)
    from zope.interface.interfaces import IInterface
    classImplements(InterfaceClass, IInterface)