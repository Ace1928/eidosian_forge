import collections
import functools
import textwrap
import warnings
from packaging import version
from datetime import date
def _function_wrapper(function):
    if should_warn:
        existing_docstring = function.__doc__ or ''
        parts = {'deprecated_in': ' %s' % deprecated_in if deprecated_in else '', 'removed_in': '\n   This will be removed {} {}.'.format('on' if isinstance(removed_in, date) else 'in', removed_in) if removed_in else '', 'details': ' %s' % details if details else ''}
        deprecation_note = '.. deprecated::{deprecated_in}{removed_in}{details}'.format(**parts)
        loc = 1
        string_list = existing_docstring.split('\n', 1)
        if len(string_list) > 1:
            string_list[1] = textwrap.dedent(string_list[1])
            string_list.insert(loc, '\n')
            if message_location != 'top':
                loc = 3
        string_list.insert(loc, deprecation_note)
        string_list.insert(loc, '\n\n')
        function.__doc__ = ''.join(string_list)

    @functools.wraps(function)
    def _inner(*args, **kwargs):
        if should_warn:
            if is_unsupported:
                cls = UnsupportedWarning
            else:
                cls = DeprecatedWarning
            the_warning = cls(function.__name__, deprecated_in, removed_in, details)
            warnings.warn(the_warning, category=DeprecationWarning, stacklevel=2)
        return function(*args, **kwargs)
    return _inner