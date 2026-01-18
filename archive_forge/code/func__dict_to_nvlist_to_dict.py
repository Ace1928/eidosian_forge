import unittest
from .._nvlist import nvlist_in, nvlist_out, _lib
from ..ctypes import (
def _dict_to_nvlist_to_dict(self, props):
    res = {}
    nv_in = nvlist_in(props)
    with nvlist_out(res) as nv_out:
        _lib.nvlist_dup(nv_in, nv_out, 0)
    return res