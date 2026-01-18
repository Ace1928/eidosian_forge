import itertools
from types import FunctionType
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface.interface import InterfaceClass
from zope.interface.interface import _decorator_non_return
from zope.interface.interface import fromFunction
def __create_class_doc(self, attrs):
    based_on = self.__abc

    def ref(c):
        mod = c.__module__
        name = c.__name__
        if mod == str.__module__:
            return '`%s`' % name
        if mod == '_io':
            mod = 'io'
        return '`{}.{}`'.format(mod, name)
    implementations_doc = '\n - '.join((ref(c) for c in sorted(self.getRegisteredConformers(), key=ref)))
    if implementations_doc:
        implementations_doc = '\n\nKnown implementations are:\n\n - ' + implementations_doc
    based_on_doc = based_on.__doc__ or ''
    based_on_doc = based_on_doc.splitlines()
    based_on_doc = based_on_doc[0] if based_on_doc else ''
    doc = 'Interface for the ABC `{}.{}`.\n\n{}{}{}'.format(based_on.__module__, based_on.__name__, attrs.get('__doc__', based_on_doc), self.__optional_methods_to_docs(attrs), implementations_doc)
    return doc