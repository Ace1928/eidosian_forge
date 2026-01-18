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
class PyAttribute(PyObj):

    def __init__(self, name=None, pyname=None, root=None, external=False, namespace='', required=False, typ=''):
        PyObj.__init__(self, name, pyname, root)
        self.required = required
        self.external = external
        self.namespace = namespace
        self.base = None
        self.type = typ
        self.fixed = False
        self.default = None

    def text(self, _target_namespace, cdict, _child=True):
        if isinstance(self.type, PyObj):
            if not cdict[self.type.name].done:
                raise MissingPrerequisite(self.type.name)
        return ([], [])

    def spec(self):
        if isinstance(self.type, PyObj):
            return f"('{self.pyname}', {self.type.class_name}, {self.required})"
        elif self.type:
            return f"('{self.pyname}', '{self.type}', {self.required})"
        else:
            return f"('{self.pyname}', '{self.base}', {self.required})"