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
def _namespace_and_tag(obj, param, top):
    try:
        namespace, tag = param.split(':')
    except ValueError:
        namespace = ''
        tag = param
    return (namespace, tag)