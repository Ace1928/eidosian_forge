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
class NULLType(Sexp, metaclass=SingletonABC):
    """A singleton class for R's NULL."""

    def __init__(self):
        if embedded.isready():
            tmp = Sexp(_rinterface.UnmanagedSexpCapsule(openrlib.rlib.R_NilValue))
        else:
            tmp = Sexp(_rinterface.UninitializedRCapsule(RTYPES.NILSXP.value))
        super().__init__(tmp)

    def __bool__(self) -> bool:
        """This is always False."""
        return False

    @property
    def __sexp__(self) -> typing.Union['_rinterface.SexpCapsule', '_rinterface.UninitializedRCapsule']:
        return self._sexpobject

    @__sexp__.setter
    def __sexp__(self, value: typing.Union['_rinterface.SexpCapsule', '_rinterface.UninitializedRCapsule']) -> None:
        raise TypeError('The capsule for the R object cannot be modified.')

    @property
    def rid(self) -> int:
        return self._sexpobject.rid