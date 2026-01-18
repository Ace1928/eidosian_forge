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
class SexpVector(Sexp, SexpVectorAbstract):
    """Base abstract class for R vector objects.

    R vector objects are, at the C level, essentially C arrays wrapped in
    the general structure for R objects."""

    def __init__(self, obj: typing.Union[SupportsSEXP, _rinterface.SexpCapsule, collections.abc.Sized]):
        if isinstance(obj, SupportsSEXP) or isinstance(obj, _rinterface.SexpCapsule):
            super().__init__(obj)
        elif isinstance(obj, collections.abc.Sized):
            robj: Sexp = type(self).from_object(obj)
            super().__init__(robj)
        else:
            raise TypeError('The constructor must be called with an instance of rpy2.rinterface.Sexp, a Python sized object that can be iterated on, or less commonly an rpy2.rinterface._rinterface.SexpCapsule.')