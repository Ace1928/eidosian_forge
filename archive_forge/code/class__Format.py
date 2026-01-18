from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
class _Format(object):

    @staticmethod
    def as_bool(args_true, args_false=None, ignore_none=None):
        if args_false is not None:
            if ignore_none is None:
                ignore_none = False
        else:
            args_false = []
        return _ArgFormat(lambda value: _ensure_list(args_true) if value else _ensure_list(args_false), ignore_none=ignore_none)

    @staticmethod
    def as_bool_not(args):
        return _ArgFormat(lambda value: [] if value else _ensure_list(args), ignore_none=False)

    @staticmethod
    def as_optval(arg, ignore_none=None):
        return _ArgFormat(lambda value: ['{0}{1}'.format(arg, value)], ignore_none=ignore_none)

    @staticmethod
    def as_opt_val(arg, ignore_none=None):
        return _ArgFormat(lambda value: [arg, value], ignore_none=ignore_none)

    @staticmethod
    def as_opt_eq_val(arg, ignore_none=None):
        return _ArgFormat(lambda value: ['{0}={1}'.format(arg, value)], ignore_none=ignore_none)

    @staticmethod
    def as_list(ignore_none=None):
        return _ArgFormat(_ensure_list, ignore_none=ignore_none)

    @staticmethod
    def as_fixed(args):
        return _ArgFormat(lambda value: _ensure_list(args), ignore_none=False, ignore_missing_value=True)

    @staticmethod
    def as_func(func, ignore_none=None):
        return _ArgFormat(func, ignore_none=ignore_none)

    @staticmethod
    def as_map(_map, default=None, ignore_none=None):
        if default is None:
            default = []
        return _ArgFormat(lambda value: _ensure_list(_map.get(value, default)), ignore_none=ignore_none)

    @staticmethod
    def as_default_type(_type, arg='', ignore_none=None):
        fmt = _Format
        if _type == 'dict':
            return fmt.as_func(lambda d: ['--{0}={1}'.format(*a) for a in iteritems(d)], ignore_none=ignore_none)
        if _type == 'list':
            return fmt.as_func(lambda value: ['--{0}'.format(x) for x in value], ignore_none=ignore_none)
        if _type == 'bool':
            return fmt.as_bool('--{0}'.format(arg))
        return fmt.as_opt_val('--{0}'.format(arg), ignore_none=ignore_none)

    @staticmethod
    def unpack_args(func):

        @wraps(func)
        def wrapper(v):
            return func(*v)
        return wrapper

    @staticmethod
    def unpack_kwargs(func):

        @wraps(func)
        def wrapper(v):
            return func(**v)
        return wrapper