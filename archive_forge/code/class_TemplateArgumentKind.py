from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class TemplateArgumentKind(BaseEnumeration):
    """
    A TemplateArgumentKind describes the kind of entity that a template argument
    represents.
    """
    _kinds = []
    _name_map = None