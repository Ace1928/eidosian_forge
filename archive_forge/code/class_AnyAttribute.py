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
class AnyAttribute(Simple):

    def repr(self, _top=None, _sup=None, _argv=None, _child=True, _parent=''):
        return PyAny()