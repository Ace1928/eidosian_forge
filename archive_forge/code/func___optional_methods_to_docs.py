import itertools
from types import FunctionType
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface.interface import InterfaceClass
from zope.interface.interface import _decorator_non_return
from zope.interface.interface import fromFunction
@staticmethod
def __optional_methods_to_docs(attrs):
    optionals = {k: v for k, v in attrs.items() if isinstance(v, optional)}
    for k in optionals:
        attrs[k] = _decorator_non_return
    if not optionals:
        return ''
    docs = '\n\nThe following methods are optional:\n - ' + '\n-'.join(('{}\n{}'.format(k, v.__doc__) for k, v in optionals.items()))
    return docs