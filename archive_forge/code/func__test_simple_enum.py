import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _test_simple_enum(checked_enum, simple_enum):
    """
    A function that can be used to test an enum created with :func:`_simple_enum`
    against the version created by subclassing :class:`Enum`::

        >>> from enum import Enum, _simple_enum, _test_simple_enum
        >>> @_simple_enum(Enum)
        ... class Color:
        ...     RED = auto()
        ...     GREEN = auto()
        ...     BLUE = auto()
        >>> class CheckedColor(Enum):
        ...     RED = auto()
        ...     GREEN = auto()
        ...     BLUE = auto()
        >>> _test_simple_enum(CheckedColor, Color)

    If differences are found, a :exc:`TypeError` is raised.
    """
    failed = []
    if checked_enum.__dict__ != simple_enum.__dict__:
        checked_dict = checked_enum.__dict__
        checked_keys = list(checked_dict.keys())
        simple_dict = simple_enum.__dict__
        simple_keys = list(simple_dict.keys())
        member_names = set(list(checked_enum._member_map_.keys()) + list(simple_enum._member_map_.keys()))
        for key in set(checked_keys + simple_keys):
            if key in ('__module__', '_member_map_', '_value2member_map_', '__doc__'):
                continue
            elif key in member_names:
                continue
            elif key not in simple_keys:
                failed.append('missing key: %r' % (key,))
            elif key not in checked_keys:
                failed.append('extra key:   %r' % (key,))
            else:
                checked_value = checked_dict[key]
                simple_value = simple_dict[key]
                if callable(checked_value) or isinstance(checked_value, bltns.property):
                    continue
                if key == '__doc__':
                    compressed_checked_value = checked_value.replace(' ', '').replace('\t', '')
                    compressed_simple_value = simple_value.replace(' ', '').replace('\t', '')
                    if compressed_checked_value != compressed_simple_value:
                        failed.append('%r:\n         %s\n         %s' % (key, 'checked -> %r' % (checked_value,), 'simple  -> %r' % (simple_value,)))
                elif checked_value != simple_value:
                    failed.append('%r:\n         %s\n         %s' % (key, 'checked -> %r' % (checked_value,), 'simple  -> %r' % (simple_value,)))
        failed.sort()
        for name in member_names:
            failed_member = []
            if name not in simple_keys:
                failed.append('missing member from simple enum: %r' % name)
            elif name not in checked_keys:
                failed.append('extra member in simple enum: %r' % name)
            else:
                checked_member_dict = checked_enum[name].__dict__
                checked_member_keys = list(checked_member_dict.keys())
                simple_member_dict = simple_enum[name].__dict__
                simple_member_keys = list(simple_member_dict.keys())
                for key in set(checked_member_keys + simple_member_keys):
                    if key in ('__module__', '__objclass__', '_inverted_'):
                        continue
                    elif key not in simple_member_keys:
                        failed_member.append('missing key %r not in the simple enum member %r' % (key, name))
                    elif key not in checked_member_keys:
                        failed_member.append('extra key %r in simple enum member %r' % (key, name))
                    else:
                        checked_value = checked_member_dict[key]
                        simple_value = simple_member_dict[key]
                        if checked_value != simple_value:
                            failed_member.append('%r:\n         %s\n         %s' % (key, 'checked member -> %r' % (checked_value,), 'simple member  -> %r' % (simple_value,)))
            if failed_member:
                failed.append('%r member mismatch:\n      %s' % (name, '\n      '.join(failed_member)))
        for method in ('__str__', '__repr__', '__reduce_ex__', '__format__', '__getnewargs_ex__', '__getnewargs__', '__reduce_ex__', '__reduce__'):
            if method in simple_keys and method in checked_keys:
                continue
            elif method not in simple_keys and method not in checked_keys:
                checked_method = getattr(checked_enum, method, None)
                simple_method = getattr(simple_enum, method, None)
                if hasattr(checked_method, '__func__'):
                    checked_method = checked_method.__func__
                    simple_method = simple_method.__func__
                if checked_method != simple_method:
                    failed.append('%r:  %-30s %s' % (method, 'checked -> %r' % (checked_method,), 'simple -> %r' % (simple_method,)))
            else:
                pass
    if failed:
        raise TypeError('enum mismatch:\n   %s' % '\n   '.join(failed))