import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
class BoolVector(Vector, BoolSexpVector):
    """ Vector of boolean (logical) elements
    BoolVector(seq) -> BoolVector.

    The parameter 'seq' can be an instance inheriting from
    rinterface.SexpVector, or an arbitrary Python sequence.
    In the later case, all elements in the sequence should be either
    booleans, or have a bool() representation.
    """
    NAvalue = rinterface.NA_Logical

    def __init__(self, obj):
        super().__init__(obj)
        self._add_rops()