import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
def _setup_vfuncs(cls):
    for vfunc_name, py_vfunc in cls.__dict__.items():
        if not vfunc_name.startswith('do_') or not callable(py_vfunc):
            continue
        skip_ambiguity_check = False
        vfunc_info = None
        for base in cls.__mro__:
            method = getattr(base, vfunc_name, None)
            if method is not None and isinstance(method, VFuncInfo):
                vfunc_info = method
                break
            if not hasattr(base, '__info__') or not hasattr(base.__info__, 'get_vfuncs'):
                continue
            base_name = snake_case(base.__info__.get_type_name())
            for v in base.__info__.get_vfuncs():
                if vfunc_name == 'do_%s_%s' % (base_name, v.get_name()):
                    vfunc_info = v
                    skip_ambiguity_check = True
                    break
            if vfunc_info:
                break
        if vfunc_info is None:
            vfunc_info = find_vfunc_info_in_interface(cls.__bases__, vfunc_name[len('do_'):])
        if vfunc_info is not None:
            if not skip_ambiguity_check:
                ambiguous_base = find_vfunc_conflict_in_bases(vfunc_info, cls.__bases__)
                if ambiguous_base is not None:
                    base_info = vfunc_info.get_container()
                    raise TypeError('Method %s() on class %s.%s is ambiguous with methods in base classes %s.%s and %s.%s' % (vfunc_name, cls.__info__.get_namespace(), cls.__info__.get_name(), base_info.get_namespace(), base_info.get_name(), ambiguous_base.__info__.get_namespace(), ambiguous_base.__info__.get_name()))
            hook_up_vfunc_implementation(vfunc_info, cls.__gtype__, py_vfunc)