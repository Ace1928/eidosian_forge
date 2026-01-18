import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
class MetaClassHelper(object):

    def _setup_methods(cls):
        for method_info in cls.__info__.get_methods():
            setattr(cls, method_info.__name__, method_info)

    def _setup_class_methods(cls):
        info = cls.__info__
        class_struct = info.get_class_struct()
        if class_struct is None:
            return
        for method_info in class_struct.get_methods():
            name = method_info.__name__
            if not hasattr(cls, name):
                setattr(cls, name, classmethod(method_info))

    def _setup_fields(cls):
        for field_info in cls.__info__.get_fields():
            name = field_info.get_name().replace('-', '_')
            setattr(cls, name, property(field_info.get_value, field_info.set_value))

    def _setup_constants(cls):
        for constant_info in cls.__info__.get_constants():
            name = constant_info.get_name()
            value = constant_info.get_value()
            setattr(cls, name, value)

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

    def _setup_native_vfuncs(cls):
        class_info = cls.__dict__.get('__info__')
        if class_info is None or not isinstance(class_info, ObjectInfo):
            return
        if cls.__module__ == 'gi.repository.GObject' and cls.__name__ == 'Object':
            return
        for vfunc_info in class_info.get_vfuncs():
            name = 'do_%s' % vfunc_info.__name__
            setattr(cls, name, vfunc_info)