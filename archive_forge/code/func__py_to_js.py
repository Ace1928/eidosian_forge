from .basedatatypes import Undefined
from .optional_imports import get_module
def _py_to_js(v, widget_manager):
    """
    Python -> Javascript ipywidget serializer

    This function must repalce all objects that the ipywidget library
    can't serialize natively (e.g. numpy arrays) with serializable
    representations

    Parameters
    ----------
    v
        Object to be serialized
    widget_manager
        ipywidget widget_manager (unused)

    Returns
    -------
    any
        Value that the ipywidget library can serialize natively
    """
    if isinstance(v, dict):
        return {k: _py_to_js(v, widget_manager) for k, v in v.items()}
    elif isinstance(v, (list, tuple)):
        return [_py_to_js(v, widget_manager) for v in v]
    elif np is not None and isinstance(v, np.ndarray):
        if v.ndim == 1 and v.dtype.kind in ['u', 'i', 'f'] and (v.dtype != 'int64') and (v.dtype != 'uint64'):
            return {'buffer': memoryview(v), 'dtype': str(v.dtype), 'shape': v.shape}
        else:
            return v.tolist()
    if v is Undefined:
        return '_undefined_'
    else:
        return v