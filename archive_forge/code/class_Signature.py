from __future__ import absolute_import, division, print_function
import itertools
import functools
import re
import types
from funcsigs.version import __version__
class Signature(object):
    """A Signature object represents the overall signature of a function.
    It stores a Parameter object for each parameter accepted by the
    function, as well as information specific to the function itself.

    A Signature object has the following public attributes and methods:

    * parameters : OrderedDict
        An ordered mapping of parameters' names to the corresponding
        Parameter objects (keyword-only arguments are in the same order
        as listed in `code.co_varnames`).
    * return_annotation : object
        The annotation for the return type of the function if specified.
        If the function has no annotation for its return type, this
        attribute is not set.
    * bind(*args, **kwargs) -> BoundArguments
        Creates a mapping from positional and keyword arguments to
        parameters.
    * bind_partial(*args, **kwargs) -> BoundArguments
        Creates a partial mapping from positional and keyword arguments
        to parameters (simulating 'functools.partial' behavior.)
    """
    __slots__ = ('_return_annotation', '_parameters')
    _parameter_cls = Parameter
    _bound_arguments_cls = BoundArguments
    empty = _empty

    def __init__(self, parameters=None, return_annotation=_empty, __validate_parameters__=True):
        """Constructs Signature from the given list of Parameter
        objects and 'return_annotation'.  All arguments are optional.
        """
        if parameters is None:
            params = OrderedDict()
        elif __validate_parameters__:
            params = OrderedDict()
            top_kind = _POSITIONAL_ONLY
            for idx, param in enumerate(parameters):
                kind = param.kind
                if kind < top_kind:
                    msg = 'wrong parameter order: {0} before {1}'
                    msg = msg.format(top_kind, param.kind)
                    raise ValueError(msg)
                else:
                    top_kind = kind
                name = param.name
                if name is None:
                    name = str(idx)
                    param = param.replace(name=name)
                if name in params:
                    msg = 'duplicate parameter name: {0!r}'.format(name)
                    raise ValueError(msg)
                params[name] = param
        else:
            params = OrderedDict(((param.name, param) for param in parameters))
        self._parameters = params
        self._return_annotation = return_annotation

    @classmethod
    def from_function(cls, func):
        """Constructs Signature for the given python function"""
        if not isinstance(func, types.FunctionType):
            raise TypeError('{0!r} is not a Python function'.format(func))
        Parameter = cls._parameter_cls
        func_code = func.__code__
        pos_count = func_code.co_argcount
        arg_names = func_code.co_varnames
        positional = tuple(arg_names[:pos_count])
        keyword_only_count = getattr(func_code, 'co_kwonlyargcount', 0)
        keyword_only = arg_names[pos_count:pos_count + keyword_only_count]
        annotations = getattr(func, '__annotations__', {})
        defaults = func.__defaults__
        kwdefaults = getattr(func, '__kwdefaults__', None)
        if defaults:
            pos_default_count = len(defaults)
        else:
            pos_default_count = 0
        parameters = []
        non_default_count = pos_count - pos_default_count
        for name in positional[:non_default_count]:
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_POSITIONAL_OR_KEYWORD))
        for offset, name in enumerate(positional[non_default_count:]):
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_POSITIONAL_OR_KEYWORD, default=defaults[offset]))
        if func_code.co_flags & 4:
            name = arg_names[pos_count + keyword_only_count]
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_VAR_POSITIONAL))
        for name in keyword_only:
            default = _empty
            if kwdefaults is not None:
                default = kwdefaults.get(name, _empty)
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_KEYWORD_ONLY, default=default))
        if func_code.co_flags & 8:
            index = pos_count + keyword_only_count
            if func_code.co_flags & 4:
                index += 1
            name = arg_names[index]
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation, kind=_VAR_KEYWORD))
        return cls(parameters, return_annotation=annotations.get('return', _empty), __validate_parameters__=False)

    @property
    def parameters(self):
        try:
            return types.MappingProxyType(self._parameters)
        except AttributeError:
            return OrderedDict(self._parameters.items())

    @property
    def return_annotation(self):
        return self._return_annotation

    def replace(self, parameters=_void, return_annotation=_void):
        """Creates a customized copy of the Signature.
        Pass 'parameters' and/or 'return_annotation' arguments
        to override them in the new copy.
        """
        if parameters is _void:
            parameters = self.parameters.values()
        if return_annotation is _void:
            return_annotation = self._return_annotation
        return type(self)(parameters, return_annotation=return_annotation)

    def __hash__(self):
        msg = "unhashable type: '{0}'".format(self.__class__.__name__)
        raise TypeError(msg)

    def __eq__(self, other):
        if not issubclass(type(other), Signature) or self.return_annotation != other.return_annotation or len(self.parameters) != len(other.parameters):
            return False
        other_positions = dict(((param, idx) for idx, param in enumerate(other.parameters.keys())))
        for idx, (param_name, param) in enumerate(self.parameters.items()):
            if param.kind == _KEYWORD_ONLY:
                try:
                    other_param = other.parameters[param_name]
                except KeyError:
                    return False
                else:
                    if param != other_param:
                        return False
            else:
                try:
                    other_idx = other_positions[param_name]
                except KeyError:
                    return False
                else:
                    if idx != other_idx or param != other.parameters[param_name]:
                        return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _bind(self, args, kwargs, partial=False):
        """Private method.  Don't use directly."""
        arguments = OrderedDict()
        parameters = iter(self.parameters.values())
        parameters_ex = ()
        arg_vals = iter(args)
        if partial:
            for param_name, param in self.parameters.items():
                if param._partial_kwarg and param_name not in kwargs:
                    kwargs[param_name] = param.default
        while True:
            try:
                arg_val = next(arg_vals)
            except StopIteration:
                try:
                    param = next(parameters)
                except StopIteration:
                    break
                else:
                    if param.kind == _VAR_POSITIONAL:
                        break
                    elif param.name in kwargs:
                        if param.kind == _POSITIONAL_ONLY:
                            msg = '{arg!r} parameter is positional only, but was passed as a keyword'
                            msg = msg.format(arg=param.name)
                            raise TypeError(msg)
                        parameters_ex = (param,)
                        break
                    elif param.kind == _VAR_KEYWORD or param.default is not _empty:
                        parameters_ex = (param,)
                        break
                    elif partial:
                        parameters_ex = (param,)
                        break
                    else:
                        msg = '{arg!r} parameter lacking default value'
                        msg = msg.format(arg=param.name)
                        raise TypeError(msg)
            else:
                try:
                    param = next(parameters)
                except StopIteration:
                    raise TypeError('too many positional arguments')
                else:
                    if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                        raise TypeError('too many positional arguments')
                    if param.kind == _VAR_POSITIONAL:
                        values = [arg_val]
                        values.extend(arg_vals)
                        arguments[param.name] = tuple(values)
                        break
                    if param.name in kwargs:
                        raise TypeError('multiple values for argument {arg!r}'.format(arg=param.name))
                    arguments[param.name] = arg_val
        kwargs_param = None
        for param in itertools.chain(parameters_ex, parameters):
            if param.kind == _POSITIONAL_ONLY:
                raise TypeError('{arg!r} parameter is positional only, but was passed as a keyword'.format(arg=param.name))
            if param.kind == _VAR_KEYWORD:
                kwargs_param = param
                continue
            param_name = param.name
            try:
                arg_val = kwargs.pop(param_name)
            except KeyError:
                if not partial and param.kind != _VAR_POSITIONAL and (param.default is _empty):
                    raise TypeError('{arg!r} parameter lacking default value'.format(arg=param_name))
            else:
                arguments[param_name] = arg_val
        if kwargs:
            if kwargs_param is not None:
                arguments[kwargs_param.name] = kwargs
            else:
                raise TypeError('too many keyword arguments %r' % kwargs)
        return self._bound_arguments_cls(self, arguments)

    def bind(*args, **kwargs):
        """Get a BoundArguments object, that maps the passed `args`
        and `kwargs` to the function's signature.  Raises `TypeError`
        if the passed arguments can not be bound.
        """
        return args[0]._bind(args[1:], kwargs)

    def bind_partial(self, *args, **kwargs):
        """Get a BoundArguments object, that partially maps the
        passed `args` and `kwargs` to the function's signature.
        Raises `TypeError` if the passed arguments can not be bound.
        """
        return self._bind(args, kwargs, partial=True)

    def __str__(self):
        result = []
        render_kw_only_separator = True
        for idx, param in enumerate(self.parameters.values()):
            formatted = str(param)
            kind = param.kind
            if kind == _VAR_POSITIONAL:
                render_kw_only_separator = False
            elif kind == _KEYWORD_ONLY and render_kw_only_separator:
                result.append('*')
                render_kw_only_separator = False
            result.append(formatted)
        rendered = '({0})'.format(', '.join(result))
        if self.return_annotation is not _empty:
            anno = formatannotation(self.return_annotation)
            rendered += ' -> {0}'.format(anno)
        return rendered