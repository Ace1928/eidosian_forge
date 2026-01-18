from _pydevd_bundle._debug_adapter.pydevd_schema_log import debug_exception
import json
import itertools
from functools import partial
@staticmethod
def _translate_id_to_dap(obj_id):
    if obj_id == '*':
        return '*'
    dap_id = BaseSchema._obj_id_to_dap_id.get(obj_id)
    if dap_id is None:
        dap_id = BaseSchema._obj_id_to_dap_id[obj_id] = BaseSchema._next_dap_id()
        BaseSchema._dap_id_to_obj_id[dap_id] = obj_id
    return dap_id