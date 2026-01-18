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
class Simple:

    def __init__(self, elem):
        self.default = None
        self.fixed = None
        self.xmlns_map = []
        self.name = ''
        self.type = None
        self.use = None
        self.ref = None
        self.scoped = False
        self.itemType = None
        for attribute, value in iter(elem.attrib.items()):
            self.__setattr__(attribute, value)

    def collect(self, top, sup, argv=None, parent=''):
        argv_copy = sd_copy(argv)
        rval = self.repr(top, sup, argv_copy, True, parent)
        if rval:
            return ([rval], [])
        else:
            return ([], [])

    def repr(self, _top=None, _sup=None, _argv=None, _child=True, _parent=''):
        return None

    def elements(self, _top):
        return []