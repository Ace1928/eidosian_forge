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
class RVersion(metaclass=Singleton):
    _version = None

    def __init__(self):
        assert embedded.isinitialized()
        robj = StrSexpVector(['R.version'])
        with memorymanagement.rmemory() as rmemory:
            parsed = _rinterface._parse(robj.__sexp__._cdata, 1, rmemory)
        res = baseenv['eval'](parsed)
        self._version = OrderedDict(((k, v[0]) for k, v in zip(res.names, res)))

    def __getitem__(self, k):
        return self._version[k]

    def keys(self):
        return self._version.keys()