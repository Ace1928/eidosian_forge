import abc
import collections.abc
from collections import OrderedDict
import enum
import itertools
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
from rpy2.rinterface_lib._rinterface_capi import _evaluated_promise
from rpy2.rinterface_lib._rinterface_capi import SupportsSEXP
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
from rpy2.rinterface_lib import na_values
class RTYPES(enum.IntEnum):
    """Native R types as defined in R's C API."""
    NILSXP = openrlib.rlib.NILSXP
    SYMSXP = openrlib.rlib.SYMSXP
    LISTSXP = openrlib.rlib.LISTSXP
    CLOSXP = openrlib.rlib.CLOSXP
    ENVSXP = openrlib.rlib.ENVSXP
    PROMSXP = openrlib.rlib.PROMSXP
    LANGSXP = openrlib.rlib.LANGSXP
    SPECIALSXP = openrlib.rlib.SPECIALSXP
    BUILTINSXP = openrlib.rlib.BUILTINSXP
    CHARSXP = openrlib.rlib.CHARSXP
    LGLSXP = openrlib.rlib.LGLSXP
    INTSXP = openrlib.rlib.INTSXP
    REALSXP = openrlib.rlib.REALSXP
    CPLXSXP = openrlib.rlib.CPLXSXP
    STRSXP = openrlib.rlib.STRSXP
    DOTSXP = openrlib.rlib.DOTSXP
    ANYSXP = openrlib.rlib.ANYSXP
    VECSXP = openrlib.rlib.VECSXP
    EXPRSXP = openrlib.rlib.EXPRSXP
    BCODESXP = openrlib.rlib.BCODESXP
    EXTPTRSXP = openrlib.rlib.EXTPTRSXP
    WEAKREFSXP = openrlib.rlib.WEAKREFSXP
    RAWSXP = openrlib.rlib.RAWSXP
    S4SXP = openrlib.rlib.S4SXP
    NEWSXP = openrlib.rlib.NEWSXP
    FREESXP = openrlib.rlib.FREESXP
    FUNSXP = openrlib.rlib.FUNSXP