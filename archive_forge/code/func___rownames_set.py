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
def __rownames_set(self, rn):
    if isinstance(rn, StrSexpVector):
        if len(rn) != self.nrow:
            raise ValueError('Invalid length.')
        if self.dimnames is NULL:
            dn = ListVector.from_length(2)
            dn[0] = rn
            self.do_slot_assign('dimnames', dn)
        else:
            dn = self.dimnames
            dn[0] = rn
    else:
        raise ValueError('The rownames attribute can only be an R string vector.')