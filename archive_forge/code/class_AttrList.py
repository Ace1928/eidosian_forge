from __future__ import absolute_import, division, unicode_literals
from xml.dom import minidom, Node
import weakref
from . import base
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
class AttrList(MutableMapping):

    def __init__(self, element):
        self.element = element

    def __iter__(self):
        return iter(self.element.attributes.keys())

    def __setitem__(self, name, value):
        if isinstance(name, tuple):
            raise NotImplementedError
        else:
            attr = self.element.ownerDocument.createAttribute(name)
            attr.value = value
            self.element.attributes[name] = attr

    def __len__(self):
        return len(self.element.attributes)

    def items(self):
        return list(self.element.attributes.items())

    def values(self):
        return list(self.element.attributes.values())

    def __getitem__(self, name):
        if isinstance(name, tuple):
            raise NotImplementedError
        else:
            return self.element.attributes[name].value

    def __delitem__(self, name):
        if isinstance(name, tuple):
            raise NotImplementedError
        else:
            del self.element.attributes[name]