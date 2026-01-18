import inspect
import sys
from types import FunctionType
from types import MethodType
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.exceptions import DoesNotImplement
from zope.interface.exceptions import Invalid
from zope.interface.exceptions import MultipleInvalid
from zope.interface.interface import Method
from zope.interface.interface import fromFunction
from zope.interface.interface import fromMethod
def _verify_element(iface, name, desc, candidate, vtype):
    try:
        attr = getattr(candidate, name)
    except AttributeError:
        if not isinstance(desc, Method) and vtype == 'c':
            return
        raise BrokenImplementation(iface, desc, candidate)
    if not isinstance(desc, Method):
        return
    if inspect.ismethoddescriptor(attr) or inspect.isbuiltin(attr):
        return
    if isinstance(attr, FunctionType):
        if isinstance(candidate, type) and vtype == 'c':
            meth = fromFunction(attr, iface, name=name, imlevel=1)
        else:
            meth = fromFunction(attr, iface, name=name)
    elif isinstance(attr, MethodTypes) and type(attr.__func__) is FunctionType:
        meth = fromMethod(attr, iface, name)
    elif isinstance(attr, property) and vtype == 'c':
        return
    else:
        if not callable(attr):
            raise BrokenMethodImplementation(desc, 'implementation is not a method', attr, iface, candidate)
        return
    mess = _incompat(desc.getSignatureInfo(), meth.getSignatureInfo())
    if mess:
        raise BrokenMethodImplementation(desc, mess, attr, iface, candidate)