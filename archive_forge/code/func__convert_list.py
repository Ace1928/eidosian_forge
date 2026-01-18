import os
from . import BioSeq
from . import Loader
from . import DBUtils
def _convert_list(self, lst):
    ret_lst = []
    for tuple_ in lst:
        new_tuple = self._convert_tuple(tuple_)
        ret_lst.append(new_tuple)
    return ret_lst