from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
class TemplateDef(object):

    def __init__(self, template, func_name, func_signature, body, ns, pos, bound_self=None):
        self._template = template
        self._func_name = func_name
        self._func_signature = func_signature
        self._body = body
        self._ns = ns
        self._pos = pos
        self._bound_self = bound_self

    def __repr__(self):
        return '<tempita function %s(%s) at %s:%s>' % (self._func_name, self._func_signature, self._template.name, self._pos)

    def __str__(self):
        return self()

    def __call__(self, *args, **kw):
        values = self._parse_signature(args, kw)
        ns = self._ns.copy()
        ns.update(values)
        if self._bound_self is not None:
            ns['self'] = self._bound_self
        out = []
        subdefs = {}
        self._template._interpret_codes(self._body, ns, out, subdefs)
        return ''.join(out)

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        return self.__class__(self._template, self._func_name, self._func_signature, self._body, self._ns, self._pos, bound_self=obj)

    def _parse_signature(self, args, kw):
        values = {}
        sig_args, var_args, var_kw, defaults = self._func_signature
        extra_kw = {}
        for name, value in kw.items():
            if not var_kw and name not in sig_args:
                raise TypeError('Unexpected argument %s' % name)
            if name in sig_args:
                values[sig_args] = value
            else:
                extra_kw[name] = value
        args = list(args)
        sig_args = list(sig_args)
        while args:
            while sig_args and sig_args[0] in values:
                sig_args.pop(0)
            if sig_args:
                name = sig_args.pop(0)
                values[name] = args.pop(0)
            elif var_args:
                values[var_args] = tuple(args)
                break
            else:
                raise TypeError('Extra position arguments: %s' % ', '.join([repr(v) for v in args]))
        for name, value_expr in defaults.items():
            if name not in values:
                values[name] = self._template._eval(value_expr, self._ns, self._pos)
        for name in sig_args:
            if name not in values:
                raise TypeError('Missing argument: %s' % name)
        if var_kw:
            values[var_kw] = extra_kw
        return values