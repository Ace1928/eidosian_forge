import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
class deprecate_func(_DecoratorBaseClass):
    """Decorate a deprecated function and warn when it is called.

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    Parameters
    ----------
    deprecated_version : str
        The package version when the deprecation was introduced.
    removed_version : str
        The package version in which the deprecated function will be removed.
    hint : str, optional
        A hint on how to address this deprecation,
        e.g., "Use `skimage.submodule.alternative_func` instead."

    Examples
    --------
    >>> @deprecate_func(
    ...     deprecated_version="1.0.0",
    ...     removed_version="1.2.0",
    ...     hint="Use `bar` instead."
    ... )
    ... def foo():
    ...     pass

    Calling ``foo`` will warn with::

        FutureWarning: `foo` is deprecated since version 1.0.0
        and will be removed in version 1.2.0. Use `bar` instead.
    """

    def __init__(self, *, deprecated_version, removed_version=None, hint=None):
        self.deprecated_version = deprecated_version
        self.removed_version = removed_version
        self.hint = hint

    def __call__(self, func):
        message = f'`{func.__name__}` is deprecated since version {self.deprecated_version}'
        if self.removed_version:
            message += f' and will be removed in version {self.removed_version}.'
        if self.hint:
            message += f' {self.hint.rstrip('.')}.'
        stack_rank = _count_wrappers(func)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank
            warnings.warn(message, category=FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)
        doc = f'**Deprecated:** {message}'
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + '\n\n    ' + wrapped.__doc__
        return wrapped