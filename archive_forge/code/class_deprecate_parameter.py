import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
class deprecate_parameter:
    """Deprecate a parameter of a function.

    Parameters
    ----------
    deprecated_name : str
        The name of the deprecated parameter.
    start_version : str
        The package version in which the warning was introduced.
    stop_version : str
        The package version in which the warning will be replaced by
        an error / the deprecation is completed.
    template : str, optional
        If given, this message template is used instead of the default one.
    new_name : str, optional
        If given, the default message will recommend the new parameter name and an
        error will be raised if the user uses both old and new names for the
        same parameter.
    modify_docstring : bool, optional
        If the wrapped function has a docstring, add the deprecated parameters
        to the "Other Parameters" section.
    stacklevel : int, optional
        This decorator attempts to detect the appropriate stacklevel for the
        deprecation warning automatically. If this fails, e.g., due to
        decorating a closure, you can set the stacklevel manually. The
        outermost decorator should have stacklevel 2, the next inner one
        stacklevel 3, etc.

    Notes
    -----
    Assign `DEPRECATED` as the new default value for the deprecated parameter.
    This marks the status of the parameter also in the signature and rendered
    HTML docs.

    This decorator can be stacked to deprecate more than one parameter.

    Examples
    --------
    >>> from skimage._shared.utils import deprecate_parameter, DEPRECATED
    >>> @deprecate_parameter(
    ...     "b", new_name="c", start_version="0.1", stop_version="0.3"
    ... )
    ... def foo(a, b=DEPRECATED, *, c=None):
    ...     return a, c

    Calling ``foo(1, b=2)``  will warn with::

        FutureWarning: Parameter `b` is deprecated since version 0.1 and will
        be removed in 0.3 (or later). To avoid this warning, please use the
        parameter `c` instead. For more details, see the documentation of
        `foo`.
    """
    DEPRECATED = DEPRECATED
    remove_parameter_template = 'Parameter `{deprecated_name}` is deprecated since version {deprecated_version} and will be removed in {changed_version} (or later). To avoid this warning, please do not use the parameter `{deprecated_name}`. For more details, see the documentation of `{func_name}`.'
    replace_parameter_template = 'Parameter `{deprecated_name}` is deprecated since version {deprecated_version} and will be removed in {changed_version} (or later). To avoid this warning, please use the parameter `{new_name}` instead. For more details, see the documentation of `{func_name}`.'

    def __init__(self, deprecated_name, *, start_version, stop_version, template=None, new_name=None, modify_docstring=True, stacklevel=None):
        self.deprecated_name = deprecated_name
        self.new_name = new_name
        self.template = template
        self.start_version = start_version
        self.stop_version = stop_version
        self.modify_docstring = modify_docstring
        self.stacklevel = stacklevel

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        deprecated_idx = list(parameters.keys()).index(self.deprecated_name)
        if self.new_name:
            new_idx = list(parameters.keys()).index(self.new_name)
        else:
            new_idx = False
        if parameters[self.deprecated_name].default is not DEPRECATED:
            raise RuntimeError(f'Expected `{self.deprecated_name}` to have the value {DEPRECATED!r} to indicate its status in the rendered signature.')
        if self.template is not None:
            template = self.template
        elif self.new_name is not None:
            template = self.replace_parameter_template
        else:
            template = self.remove_parameter_template
        warning_message = template.format(deprecated_name=self.deprecated_name, deprecated_version=self.start_version, changed_version=self.stop_version, func_name=func.__qualname__, new_name=self.new_name)

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            deprecated_value = DEPRECATED
            new_value = DEPRECATED
            if len(args) > deprecated_idx:
                deprecated_value = args[deprecated_idx]
                args = args[:deprecated_idx] + (DEPRECATED,) + args[deprecated_idx + 1:]
            if self.deprecated_name in kwargs.keys():
                deprecated_value = kwargs[self.deprecated_name]
                kwargs[self.deprecated_name] = DEPRECATED
            if new_idx is not False and len(args) > new_idx:
                new_value = args[new_idx]
            if self.new_name and self.new_name in kwargs.keys():
                new_value = kwargs[self.new_name]
            if deprecated_value is not DEPRECATED:
                stacklevel = self.stacklevel if self.stacklevel is not None else _warning_stacklevel(func)
                warnings.warn(warning_message, category=FutureWarning, stacklevel=stacklevel)
                if new_value is not DEPRECATED:
                    raise ValueError(f'Both deprecated parameter `{self.deprecated_name}` and new parameter `{self.new_name}` are used. Use only the latter to avoid conflicting values.')
                elif self.new_name is not None:
                    kwargs[self.new_name] = deprecated_value
            return func(*args, **kwargs)
        if self.modify_docstring and func.__doc__ is not None:
            newdoc = _docstring_add_deprecated(func, {self.deprecated_name: self.new_name}, self.start_version)
            fixed_func.__doc__ = newdoc
        return fixed_func