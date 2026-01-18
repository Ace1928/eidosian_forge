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
def _do_from_string(name):
    print
    print(f'def {pyify(name)}_from_string(xml_string):')
    print(f'{INDENT}return saml2.create_class_from_xml_string({name}, xml_string)')