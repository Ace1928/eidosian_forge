import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def _get_optional_content(self, oc: OptInt) -> OptStr:
    if oc is None or oc == 0:
        return None
    doc = self.parent
    check = doc.xref_object(oc, compressed=True)
    if not ('/Type/OCG' in check or '/Type/OCMD' in check):
        raise ValueError("bad optional content: 'oc'")
    props = {}
    for p, x in self._get_resource_properties():
        props[x] = p
    if oc in props.keys():
        return props[oc]
    i = 0
    mc = 'MC%i' % i
    while mc in props.values():
        i += 1
        mc = 'MC%i' % i
    self._set_resource_property(mc, oc)
    return mc