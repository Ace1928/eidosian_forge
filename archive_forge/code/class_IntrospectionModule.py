import importlib
from threading import Lock
import gi
from ._gi import \
from .types import \
from ._constants import \
class IntrospectionModule(object):
    """An object which wraps an introspection typelib.

    This wrapping creates a python module like representation of the typelib
    using gi repository as a foundation. Accessing attributes of the module
    will dynamically pull them in and create wrappers for the members.
    These members are then cached on this introspection module.
    """

    def __init__(self, namespace, version=None):
        """Might raise gi._gi.RepositoryError"""
        repository.require(namespace, version)
        self._namespace = namespace
        self._version = version
        self.__name__ = 'gi.repository.' + namespace
        path = repository.get_typelib_path(self._namespace)
        self.__path__ = [path]
        if self._version is None:
            self._version = repository.get_version(self._namespace)
        self._lock = Lock()

    def __getattr__(self, name):
        info = repository.find_by_name(self._namespace, name)
        if not info:
            raise AttributeError('%r object has no attribute %r' % (self.__name__, name))
        if isinstance(info, EnumInfo):
            g_type = info.get_g_type()
            with self._lock:
                wrapper = g_type.pytype
                if wrapper is None:
                    if info.is_flags():
                        if g_type.is_a(TYPE_FLAGS):
                            wrapper = flags_add(g_type)
                        else:
                            assert g_type == TYPE_NONE
                            wrapper = flags_register_new_gtype_and_add(info)
                    elif g_type.is_a(TYPE_ENUM):
                        wrapper = enum_add(g_type)
                    else:
                        assert g_type == TYPE_NONE
                        wrapper = enum_register_new_gtype_and_add(info)
                    wrapper.__info__ = info
                    wrapper.__module__ = 'gi.repository.' + info.get_namespace()
                    ascii_upper_trans = ''.maketrans('abcdefgjhijklmnopqrstuvwxyz', 'ABCDEFGJHIJKLMNOPQRSTUVWXYZ')
                    for value_info in info.get_values():
                        value_name = value_info.get_name_unescaped().translate(ascii_upper_trans)
                        setattr(wrapper, value_name, wrapper(value_info.get_value()))
                    for method_info in info.get_methods():
                        setattr(wrapper, method_info.__name__, method_info)
                if g_type != TYPE_NONE:
                    g_type.pytype = wrapper
        elif isinstance(info, RegisteredTypeInfo):
            g_type = info.get_g_type()
            if isinstance(info, ObjectInfo):
                parent = get_parent_for_object(info)
                interfaces = tuple((interface for interface in get_interfaces_for_object(info) if not issubclass(parent, interface)))
                bases = (parent,) + interfaces
                metaclass = GObjectMeta
            elif isinstance(info, CallbackInfo):
                bases = (CCallback,)
                metaclass = GObjectMeta
            elif isinstance(info, InterfaceInfo):
                bases = (GInterface,)
                metaclass = GObjectMeta
            elif isinstance(info, (StructInfo, UnionInfo)):
                if g_type.is_a(TYPE_BOXED):
                    bases = (Boxed,)
                elif g_type.is_a(TYPE_POINTER) or g_type == TYPE_NONE or g_type.fundamental == g_type:
                    bases = (Struct,)
                else:
                    raise TypeError('unable to create a wrapper for %s.%s' % (info.get_namespace(), info.get_name()))
                metaclass = StructMeta
            else:
                raise NotImplementedError(info)
            with self._lock:
                if g_type != TYPE_NONE:
                    type_ = g_type.pytype
                    if type_ is not None and type_ not in bases:
                        self.__dict__[name] = type_
                        return type_
                dict_ = {'__info__': info, '__module__': 'gi.repository.' + self._namespace, '__gtype__': g_type}
                wrapper = metaclass(name, bases, dict_)
                if g_type != TYPE_NONE:
                    g_type.pytype = wrapper
        elif isinstance(info, FunctionInfo):
            wrapper = info
        elif isinstance(info, ConstantInfo):
            wrapper = info.get_value()
        else:
            raise NotImplementedError(info)
        self.__dict__[name] = wrapper
        return wrapper

    def __repr__(self):
        path = repository.get_typelib_path(self._namespace)
        return '<IntrospectionModule %r from %r>' % (self._namespace, path)

    def __dir__(self):
        result = set(dir(self.__class__))
        result.update(self.__dict__.keys())
        namespace_infos = repository.get_infos(self._namespace)
        result.update((info.get_name() for info in namespace_infos if not isinstance(info, CallbackInfo)))
        return list(result)