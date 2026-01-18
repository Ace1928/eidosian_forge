import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
def _typedef(obj, derive=False, frames=False, infer=False):
    """Create a new typedef for an object."""
    t = type(obj)
    v = _Typedef(base=_basicsize(t, obj=obj), kind=_kind_dynamic, type=t)
    if ismodule(obj):
        v.dup(item=_dict_typedef.item + _sizeof_CPyModuleObject, leng=_len_module, refs=_module_refs)
    elif _isframe(obj):
        v.set(base=_basicsize(t, base=_sizeof_CPyFrameObject, obj=obj), item=_itemsize(t), leng=_len_frame, refs=_frame_refs)
        if not frames:
            v.set(kind=_kind_ignored)
    elif iscode(obj):
        v.set(base=_basicsize(t, base=_sizeof_CPyCodeObject, obj=obj), item=_sizeof_Cvoidp, leng=_len_code, refs=_co_refs, both=False)
    elif callable(obj):
        if isclass(obj):
            v.set(refs=_class_refs, both=False)
            if _isignored(obj):
                v.set(kind=_kind_ignored)
        elif isbuiltin(obj):
            v.set(both=False, kind=_kind_ignored)
        elif isfunction(obj):
            v.set(refs=_func_refs, both=False)
        elif ismethod(obj):
            v.set(refs=_im_refs, both=False)
        elif isclass(t):
            v.set(item=_itemsize(t), safe_len=True, refs=_inst_refs)
        else:
            v.set(both=False)
    elif _issubclass(t, dict):
        v.dup(kind=_kind_derived)
    elif _isdictype(obj) or (infer and _infer_dict(obj)):
        v.dup(kind=_kind_inferred)
    elif _iscell(obj):
        v.set(item=_itemsize(t), refs=_cell_refs)
    elif _isnamedtuple(obj):
        v.set(refs=_namedtuple_refs)
    elif _numpy and _isnumpy(obj):
        v.set(**_numpy_kwds(obj))
    elif isinstance(obj, _array):
        v.set(**_array_kwds(obj))
    elif _isignored(obj):
        v.set(kind=_kind_ignored)
    else:
        if derive:
            p = _derive_typedef(t)
            if p:
                v.dup(other=p, kind=_kind_derived)
                return v
        if _issubclass(t, Exception):
            v.set(item=_itemsize(t), safe_len=True, refs=_exc_refs, kind=_kind_derived)
        elif isinstance(obj, Exception):
            v.set(item=_itemsize(t), safe_len=True, refs=_exc_refs)
        else:
            v.set(item=_itemsize(t), safe_len=True, refs=_inst_refs)
    return v