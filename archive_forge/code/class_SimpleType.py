import errno
import getopt
import importlib
import re
import sys
import time
import types
from xml.etree import ElementTree as ElementTree
import saml2
from saml2 import SamlBase
class SimpleType(Complex):

    def repr(self, top=None, _sup=None, _argv=None, _child=True, parent=''):
        if self.py_class:
            return self.py_class
        obj = PyType(self.name, root=top)
        try:
            if len(self.parts) == 1:
                part = self.parts[0]
                if isinstance(part, Restriction):
                    if part.parts:
                        if isinstance(part.parts[0], Enumeration):
                            lista = [p.value for p in part.parts]
                            obj.value_type = {'base': part.base, 'enumeration': lista}
                        elif isinstance(part.parts[0], MaxLength):
                            obj.value_type = {'base': part.base, 'maxlen': part.parts[0].value}
                        elif isinstance(part.parts[0], Length):
                            obj.value_type = {'base': part.base, 'len': part.parts[0].value}
                    else:
                        obj.value_type = {'base': part.base}
                elif isinstance(part, List):
                    if part.itemType:
                        obj.value_type = {'base': 'list', 'member': part.itemType}
        except ValueError:
            pass
        self.py_class = obj
        return obj