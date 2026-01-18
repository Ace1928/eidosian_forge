import inspect
import os
import re
import textwrap
import typing
from typing import Union
import warnings
from collections import OrderedDict
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp
from rpy2.robjects import help
from rpy2.robjects import conversion
from rpy2.robjects.vectors import Vector
from rpy2.robjects.packages_utils import (default_symbol_r2python,
class DocstringProperty(object):

    def __init__(self, class_doc, fget):
        self.fget = fget
        self.class_doc = class_doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.class_doc
        else:
            return self.fget(obj)

    def __set__(self, obj, value):
        raise AttributeError('Cannot set the attribute')

    def __delete__(self, obj):
        raise AttributeError('Cannot delete the attribute')