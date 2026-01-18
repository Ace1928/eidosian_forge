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
def _formals_fixed(func):
    if func.typeof in (rpy2.rinterface_lib.sexp.RTYPES.SPECIALSXP, rpy2.rinterface_lib.sexp.RTYPES.BUILTINSXP):
        res = rpy2.rinterface_lib.sexp.NULL
    else:
        res = __formals(func)
        if res is rpy2.rinterface_lib.sexp.NULL:
            res = __formals(__args(func))
    return res