import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
class DirectObjectAccess:

    def __init__(self, inference_state, obj):
        self._inference_state = inference_state
        self._obj = obj

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.get_repr())

    def _create_access(self, obj):
        return create_access(self._inference_state, obj)

    def _create_access_path(self, obj) -> AccessPath:
        return create_access_path(self._inference_state, obj)

    def py__bool__(self):
        return bool(self._obj)

    def py__file__(self) -> Optional[Path]:
        try:
            return Path(self._obj.__file__)
        except AttributeError:
            return None

    def py__doc__(self):
        return inspect.getdoc(self._obj) or ''

    def py__name__(self):
        if not _is_class_instance(self._obj) or inspect.ismethoddescriptor(self._obj):
            cls = self._obj
        else:
            try:
                cls = self._obj.__class__
            except AttributeError:
                return None
        try:
            return cls.__name__
        except AttributeError:
            return None

    def py__mro__accesses(self):
        return tuple((self._create_access_path(cls) for cls in self._obj.__mro__[1:]))

    def py__getitem__all_values(self):
        if isinstance(self._obj, dict):
            return [self._create_access_path(v) for v in self._obj.values()]
        if isinstance(self._obj, (list, tuple)):
            return [self._create_access_path(v) for v in self._obj]
        if self.is_instance():
            cls = DirectObjectAccess(self._inference_state, self._obj.__class__)
            return cls.py__getitem__all_values()
        try:
            getitem = self._obj.__getitem__
        except AttributeError:
            pass
        else:
            annotation = DirectObjectAccess(self._inference_state, getitem).get_return_annotation()
            if annotation is not None:
                return [annotation]
        return None

    def py__simple_getitem__(self, index, *, safe=True):
        if safe and type(self._obj) not in ALLOWED_GETITEM_TYPES:
            return None
        return self._create_access_path(self._obj[index])

    def py__iter__list(self):
        try:
            iter_method = self._obj.__iter__
        except AttributeError:
            return None
        else:
            p = DirectObjectAccess(self._inference_state, iter_method).get_return_annotation()
            if p is not None:
                return [p]
        if type(self._obj) not in ALLOWED_GETITEM_TYPES:
            return []
        lst = []
        for i, part in enumerate(self._obj):
            if i > 20:
                break
            lst.append(self._create_access_path(part))
        return lst

    def py__class__(self):
        return self._create_access_path(self._obj.__class__)

    def py__bases__(self):
        return [self._create_access_path(base) for base in self._obj.__bases__]

    def py__path__(self):
        paths = getattr(self._obj, '__path__', None)
        if not isinstance(paths, list) or not all((isinstance(p, str) for p in paths)):
            return None
        return paths

    @shorten_repr
    def get_repr(self):
        if inspect.ismodule(self._obj):
            return repr(self._obj)
        if safe_getattr(self._obj, '__module__', default='') == 'builtins':
            return repr(self._obj)
        type_ = type(self._obj)
        if type_ == type:
            return type.__repr__(self._obj)
        if safe_getattr(type_, '__module__', default='') == 'builtins':
            return repr(self._obj)
        return object.__repr__(self._obj)

    def is_class(self):
        return inspect.isclass(self._obj)

    def is_function(self):
        return inspect.isfunction(self._obj) or inspect.ismethod(self._obj)

    def is_module(self):
        return inspect.ismodule(self._obj)

    def is_instance(self):
        return _is_class_instance(self._obj)

    def ismethoddescriptor(self):
        return inspect.ismethoddescriptor(self._obj)

    def get_qualified_names(self):

        def try_to_get_name(obj):
            return getattr(obj, '__qualname__', getattr(obj, '__name__', None))
        if self.is_module():
            return ()
        name = try_to_get_name(self._obj)
        if name is None:
            name = try_to_get_name(type(self._obj))
            if name is None:
                return ()
        return tuple(name.split('.'))

    def dir(self):
        return dir(self._obj)

    def has_iter(self):
        try:
            iter(self._obj)
            return True
        except TypeError:
            return False

    def is_allowed_getattr(self, name, safe=True) -> Tuple[bool, bool, Optional[AccessPath]]:
        try:
            attr, is_get_descriptor = getattr_static(self._obj, name)
        except AttributeError:
            if not safe:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('always')
                    try:
                        return (hasattr(self._obj, name), False, None)
                    except Exception:
                        pass
            return (False, False, None)
        else:
            if is_get_descriptor and type(attr) not in ALLOWED_DESCRIPTOR_ACCESS:
                if isinstance(attr, property):
                    if hasattr(attr.fget, '__annotations__'):
                        a = DirectObjectAccess(self._inference_state, attr.fget)
                        return (True, True, a.get_return_annotation())
                return (True, True, None)
        return (True, False, None)

    def getattr_paths(self, name, default=_sentinel):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                return_obj = getattr(self._obj, name)
        except Exception as e:
            if default is _sentinel:
                if isinstance(e, AttributeError):
                    raise
                raise AttributeError
            return_obj = default
        access = self._create_access(return_obj)
        if inspect.ismodule(return_obj):
            return [access]
        try:
            module = return_obj.__module__
        except AttributeError:
            pass
        else:
            if module is not None and isinstance(module, str):
                try:
                    __import__(module)
                except ImportError:
                    pass
        module = inspect.getmodule(return_obj)
        if module is None:
            module = inspect.getmodule(type(return_obj))
            if module is None:
                module = builtins
        return [self._create_access(module), access]

    def get_safe_value(self):
        if type(self._obj) in (bool, bytes, float, int, str, slice) or self._obj is None:
            return self._obj
        raise ValueError('Object is type %s and not simple' % type(self._obj))

    def get_api_type(self):
        return get_api_type(self._obj)

    def get_array_type(self):
        if isinstance(self._obj, dict):
            return 'dict'
        return None

    def get_key_paths(self):

        def iter_partial_keys():
            for i, k in enumerate(self._obj.keys()):
                if i > 50:
                    break
                yield k
        return [self._create_access_path(k) for k in iter_partial_keys()]

    def get_access_path_tuples(self):
        accesses = [create_access(self._inference_state, o) for o in self._get_objects_path()]
        return [(access.py__name__(), access) for access in accesses]

    def _get_objects_path(self):

        def get():
            obj = self._obj
            yield obj
            try:
                obj = obj.__objclass__
            except AttributeError:
                pass
            else:
                yield obj
            try:
                imp_plz = obj.__module__
            except AttributeError:
                if not inspect.ismodule(obj):
                    yield builtins
            else:
                if imp_plz is None:
                    yield builtins
                else:
                    try:
                        yield sys.modules[imp_plz]
                    except KeyError:
                        yield builtins
        return list(reversed(list(get())))

    def execute_operation(self, other_access_handle, operator):
        other_access = other_access_handle.access
        op = _OPERATORS[operator]
        return self._create_access_path(op(self._obj, other_access._obj))

    def get_annotation_name_and_args(self):
        """
        Returns Tuple[Optional[str], Tuple[AccessPath, ...]]
        """
        name = None
        args = ()
        if safe_getattr(self._obj, '__module__', default='') == 'typing':
            m = re.match('typing.(\\w+)\\[', repr(self._obj))
            if m is not None:
                name = m.group(1)
                import typing
                if sys.version_info >= (3, 8):
                    args = typing.get_args(self._obj)
                else:
                    args = safe_getattr(self._obj, '__args__', default=None)
        return (name, tuple((self._create_access_path(arg) for arg in args)))

    def needs_type_completions(self):
        return inspect.isclass(self._obj) and self._obj != type

    def _annotation_to_str(self, annotation):
        return inspect.formatannotation(annotation)

    def get_signature_params(self):
        return [SignatureParam(name=p.name, has_default=p.default is not p.empty, default=self._create_access_path(p.default), default_string=repr(p.default), has_annotation=p.annotation is not p.empty, annotation=self._create_access_path(p.annotation), annotation_string=self._annotation_to_str(p.annotation), kind_name=str(p.kind)) for p in self._get_signature().parameters.values()]

    def _get_signature(self):
        obj = self._obj
        try:
            return inspect.signature(obj)
        except (RuntimeError, TypeError):
            raise ValueError

    def get_return_annotation(self) -> Optional[AccessPath]:
        try:
            o = self._obj.__annotations__.get('return')
        except AttributeError:
            return None
        if o is None:
            return None
        try:
            o = typing.get_type_hints(self._obj).get('return')
        except Exception:
            pass
        return self._create_access_path(o)

    def negate(self):
        return self._create_access_path(-self._obj)

    def get_dir_infos(self):
        """
        Used to return a couple of infos that are needed when accessing the sub
        objects of an objects
        """
        tuples = dict(((name, self.is_allowed_getattr(name)) for name in self.dir()))
        return (self.needs_type_completions(), tuples)