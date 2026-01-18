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
@classmethod
def _raise_incompatible_C_size(cls, mview):
    msg = 'Incompatible C type sizes. The R array type is "{r_type}" with {r_size} byte{r_size_pl} per item while the Python array type is "{py_type}" with {py_size} byte{py_size_pl} per item.'.format(r_type=cls._R_TYPE, r_size=cls._R_SIZEOF_ELT, r_size_pl='s' if cls._R_SIZEOF_ELT > 1 else '', py_type=mview.format, py_size=mview.itemsize, py_size_pl='s' if mview.itemsize > 1 else '')
    raise ValueError(msg)