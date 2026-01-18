from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
class CharacterAttributesMixin:
    """
    Mixin for character attributes that implements a C{__getattr__} method
    returning a new C{_NormalAttr} instance when attempting to access
    a C{'normal'} attribute; otherwise a new C{_OtherAttr} instance is returned
    for names that appears in the C{'attrs'} attribute.
    """

    def __getattr__(self, name):
        if name == 'normal':
            return _NormalAttr()
        if name in self.attrs:
            return _OtherAttr(name, True)
        raise AttributeError(name)