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
def _mod_typ(prop):
    try:
        mod, typ = prop.type
    except ValueError:
        typ = prop.type
        mod = None
    except TypeError:
        try:
            mod, typ = prop.ref
        except ValueError:
            if prop.class_name:
                typ = prop.class_name
            else:
                typ = prop.ref
            mod = None
    return (mod, typ)