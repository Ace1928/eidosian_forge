from __future__ import print_function, absolute_import
import sys
import os
import traceback
import types
import signature_bootstrap
from shibokensupport import signature
import shibokensupport
from shibokensupport.signature import mapping
from shibokensupport.signature import errorhandler
from shibokensupport.signature import layout
from shibokensupport.signature import lib
from shibokensupport.signature import parser
from shibokensupport.signature.lib import enum_sig
from shibokensupport.signature.parser import pyside_type_init
def _set_modname(mod, name):
    if getattr(mod, '__spec__', None):
        mod.__spec__.name = name
    else:
        mod.__name__ = name