from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
@classmethod
def _get_function_info(cls, func_ir):
    """
        Returns
        -------
        qualname, unique_name, modname, doc, args, kws, globals

        ``unique_name`` must be a unique name.
        """
    func = func_ir.func_id.func
    qualname = func_ir.func_id.func_qualname
    modname = func.__module__
    doc = func.__doc__ or ''
    args = tuple(func_ir.arg_names)
    kws = ()
    global_dict = None
    if modname is None:
        modname = _dynamic_modname
        global_dict = func_ir.func_id.func.__globals__
    unique_name = func_ir.func_id.unique_name
    return (qualname, unique_name, modname, doc, args, kws, global_dict)