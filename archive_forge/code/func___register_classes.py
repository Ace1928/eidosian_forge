import itertools
from types import FunctionType
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface.interface import InterfaceClass
from zope.interface.interface import _decorator_non_return
from zope.interface.interface import fromFunction
def __register_classes(self, conformers=None, ignored_classes=None):
    conformers = conformers if conformers is not None else self.getRegisteredConformers()
    ignored = ignored_classes if ignored_classes is not None else self.__ignored_classes
    for cls in conformers:
        if cls in ignored:
            continue
        classImplements(cls, self)