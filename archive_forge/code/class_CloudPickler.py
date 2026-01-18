import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
class CloudPickler(Pickler):
    _dispatch_table = {}
    _dispatch_table[classmethod] = _classmethod_reduce
    _dispatch_table[io.TextIOWrapper] = _file_reduce
    _dispatch_table[logging.Logger] = _logger_reduce
    _dispatch_table[logging.RootLogger] = _root_logger_reduce
    _dispatch_table[memoryview] = _memoryview_reduce
    _dispatch_table[property] = _property_reduce
    _dispatch_table[staticmethod] = _classmethod_reduce
    _dispatch_table[CellType] = _cell_reduce
    _dispatch_table[types.CodeType] = _code_reduce
    _dispatch_table[types.GetSetDescriptorType] = _getset_descriptor_reduce
    _dispatch_table[types.ModuleType] = _module_reduce
    _dispatch_table[types.MethodType] = _method_reduce
    _dispatch_table[types.MappingProxyType] = _mappingproxy_reduce
    _dispatch_table[weakref.WeakSet] = _weakset_reduce
    _dispatch_table[typing.TypeVar] = _typevar_reduce
    _dispatch_table[_collections_abc.dict_keys] = _dict_keys_reduce
    _dispatch_table[_collections_abc.dict_values] = _dict_values_reduce
    _dispatch_table[_collections_abc.dict_items] = _dict_items_reduce
    _dispatch_table[type(OrderedDict().keys())] = _odict_keys_reduce
    _dispatch_table[type(OrderedDict().values())] = _odict_values_reduce
    _dispatch_table[type(OrderedDict().items())] = _odict_items_reduce
    _dispatch_table[abc.abstractmethod] = _classmethod_reduce
    _dispatch_table[abc.abstractclassmethod] = _classmethod_reduce
    _dispatch_table[abc.abstractstaticmethod] = _classmethod_reduce
    _dispatch_table[abc.abstractproperty] = _property_reduce
    dispatch_table = ChainMap(_dispatch_table, copyreg.dispatch_table)

    def _dynamic_function_reduce(self, func):
        """Reduce a function that is not pickleable via attribute lookup."""
        newargs = self._function_getnewargs(func)
        state = _function_getstate(func)
        return (_make_function, newargs, state, None, None, _function_setstate)

    def _function_reduce(self, obj):
        """Reducer for function objects.

        If obj is a top-level attribute of a file-backed module, this
        reducer returns NotImplemented, making the CloudPickler fallback to
        traditional _pickle.Pickler routines to save obj. Otherwise, it reduces
        obj using a custom cloudpickle reducer designed specifically to handle
        dynamic functions.

        As opposed to cloudpickle.py, There no special handling for builtin
        pypy functions because cloudpickle_fast is CPython-specific.
        """
        if _should_pickle_by_reference(obj):
            return NotImplemented
        else:
            return self._dynamic_function_reduce(obj)

    def _function_getnewargs(self, func):
        code = func.__code__
        base_globals = self.globals_ref.setdefault(id(func.__globals__), {})
        if base_globals == {}:
            for k in ['__package__', '__name__', '__path__', '__file__']:
                if k in func.__globals__:
                    base_globals[k] = func.__globals__[k]
        if func.__closure__ is None:
            closure = None
        else:
            closure = tuple((_make_empty_cell() for _ in range(len(code.co_freevars))))
        return (code, base_globals, None, None, closure)

    def dump(self, obj):
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if 'recursion' in e.args[0]:
                msg = 'Could not pickle object as excessively deep recursion required.'
                raise pickle.PicklingError(msg) from e
            else:
                raise
    if pickle.HIGHEST_PROTOCOL >= 5:

        def __init__(self, file, protocol=None, buffer_callback=None):
            if protocol is None:
                protocol = DEFAULT_PROTOCOL
            Pickler.__init__(self, file, protocol=protocol, buffer_callback=buffer_callback)
            self.globals_ref = {}
            self.proto = int(protocol)
    else:

        def __init__(self, file, protocol=None):
            if protocol is None:
                protocol = DEFAULT_PROTOCOL
            Pickler.__init__(self, file, protocol=protocol)
            self.globals_ref = {}
            assert hasattr(self, 'proto')
    if pickle.HIGHEST_PROTOCOL >= 5 and (not PYPY):
        dispatch = dispatch_table

        def reducer_override(self, obj):
            """Type-agnostic reducing callback for function and classes.

            For performance reasons, subclasses of the C _pickle.Pickler class
            cannot register custom reducers for functions and classes in the
            dispatch_table. Reducer for such types must instead implemented in
            the special reducer_override method.

            Note that method will be called for any object except a few
            builtin-types (int, lists, dicts etc.), which differs from reducers
            in the Pickler's dispatch_table, each of them being invoked for
            objects of a specific type only.

            This property comes in handy for classes: although most classes are
            instances of the ``type`` metaclass, some of them can be instances
            of other custom metaclasses (such as enum.EnumMeta for example). In
            particular, the metaclass will likely not be known in advance, and
            thus cannot be special-cased using an entry in the dispatch_table.
            reducer_override, among other things, allows us to register a
            reducer that will be called for any class, independently of its
            type.


            Notes:

            * reducer_override has the priority over dispatch_table-registered
            reducers.
            * reducer_override can be used to fix other limitations of
              cloudpickle for other types that suffered from type-specific
              reducers, such as Exceptions. See
              https://github.com/cloudpipe/cloudpickle/issues/248
            """
            if sys.version_info[:2] < (3, 7) and _is_parametrized_type_hint(obj):
                return (_create_parametrized_type_hint, parametrized_type_hint_getinitargs(obj))
            t = type(obj)
            try:
                is_anyclass = issubclass(t, type)
            except TypeError:
                is_anyclass = False
            if is_anyclass:
                return _class_reduce(obj)
            elif isinstance(obj, types.FunctionType):
                return self._function_reduce(obj)
            else:
                return NotImplemented
    else:
        dispatch = Pickler.dispatch.copy()

        def _save_reduce_pickle5(self, func, args, state=None, listitems=None, dictitems=None, state_setter=None, obj=None):
            save = self.save
            write = self.write
            self.save_reduce(func, args, state=None, listitems=listitems, dictitems=dictitems, obj=obj)
            save(state_setter)
            save(obj)
            save(state)
            write(pickle.TUPLE2)
            write(pickle.REDUCE)
            write(pickle.POP)

        def save_global(self, obj, name=None, pack=struct.pack):
            """
            Save a "global".

            The name of this method is somewhat misleading: all types get
            dispatched here.
            """
            if obj is type(None):
                return self.save_reduce(type, (None,), obj=obj)
            elif obj is type(Ellipsis):
                return self.save_reduce(type, (Ellipsis,), obj=obj)
            elif obj is type(NotImplemented):
                return self.save_reduce(type, (NotImplemented,), obj=obj)
            elif obj in _BUILTIN_TYPE_NAMES:
                return self.save_reduce(_builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)
            if sys.version_info[:2] < (3, 7) and _is_parametrized_type_hint(obj):
                self.save_reduce(_create_parametrized_type_hint, parametrized_type_hint_getinitargs(obj), obj=obj)
            elif name is not None:
                Pickler.save_global(self, obj, name=name)
            elif not _should_pickle_by_reference(obj, name=name):
                self._save_reduce_pickle5(*_dynamic_class_reduce(obj), obj=obj)
            else:
                Pickler.save_global(self, obj, name=name)
        dispatch[type] = save_global

        def save_function(self, obj, name=None):
            """Registered with the dispatch to handle all function types.

            Determines what kind of function obj is (e.g. lambda, defined at
            interactive prompt, etc) and handles the pickling appropriately.
            """
            if _should_pickle_by_reference(obj, name=name):
                return Pickler.save_global(self, obj, name=name)
            elif PYPY and isinstance(obj.__code__, builtin_code_type):
                return self.save_pypy_builtin_func(obj)
            else:
                return self._save_reduce_pickle5(*self._dynamic_function_reduce(obj), obj=obj)

        def save_pypy_builtin_func(self, obj):
            """Save pypy equivalent of builtin functions.
            PyPy does not have the concept of builtin-functions. Instead,
            builtin-functions are simple function instances, but with a
            builtin-code attribute.
            Most of the time, builtin functions should be pickled by attribute.
            But PyPy has flaky support for __qualname__, so some builtin
            functions such as float.__new__ will be classified as dynamic. For
            this reason only, we created this special routine. Because
            builtin-functions are not expected to have closure or globals,
            there is no additional hack (compared the one already implemented
            in pickle) to protect ourselves from reference cycles. A simple
            (reconstructor, newargs, obj.__dict__) tuple is save_reduced.  Note
            also that PyPy improved their support for __qualname__ in v3.6, so
            this routing should be removed when cloudpickle supports only PyPy
            3.6 and later.
            """
            rv = (types.FunctionType, (obj.__code__, {}, obj.__name__, obj.__defaults__, obj.__closure__), obj.__dict__)
            self.save_reduce(*rv, obj=obj)
        dispatch[types.FunctionType] = save_function