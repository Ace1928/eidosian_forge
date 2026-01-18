import inspect
import logging
import re
from enum import Enum
from django.template.context import BaseContext
from django.utils.formats import localize
from django.utils.html import conditional_escape, escape
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import get_text_list, smart_split, unescape_string_literal
from django.utils.timezone import template_localtime
from django.utils.translation import gettext_lazy, pgettext_lazy
from .exceptions import TemplateSyntaxError
class FilterExpression:
    """
    Parse a variable token and its optional filters (all as a single string),
    and return a list of tuples of the filter name and arguments.
    Sample::

        >>> token = 'variable|default:"Default value"|date:"Y-m-d"'
        >>> p = Parser('')
        >>> fe = FilterExpression(token, p)
        >>> len(fe.filters)
        2
        >>> fe.var
        <Variable: 'variable'>
    """
    __slots__ = ('token', 'filters', 'var', 'is_var')

    def __init__(self, token, parser):
        self.token = token
        matches = filter_re.finditer(token)
        var_obj = None
        filters = []
        upto = 0
        for match in matches:
            start = match.start()
            if upto != start:
                raise TemplateSyntaxError('Could not parse some characters: %s|%s|%s' % (token[:upto], token[upto:start], token[start:]))
            if var_obj is None:
                if (constant := match['constant']):
                    try:
                        var_obj = Variable(constant).resolve({})
                    except VariableDoesNotExist:
                        var_obj = None
                elif (var := match['var']) is None:
                    raise TemplateSyntaxError('Could not find variable at start of %s.' % token)
                else:
                    var_obj = Variable(var)
            else:
                filter_name = match['filter_name']
                args = []
                if (constant_arg := match['constant_arg']):
                    args.append((False, Variable(constant_arg).resolve({})))
                elif (var_arg := match['var_arg']):
                    args.append((True, Variable(var_arg)))
                filter_func = parser.find_filter(filter_name)
                self.args_check(filter_name, filter_func, args)
                filters.append((filter_func, args))
            upto = match.end()
        if upto != len(token):
            raise TemplateSyntaxError("Could not parse the remainder: '%s' from '%s'" % (token[upto:], token))
        self.filters = filters
        self.var = var_obj
        self.is_var = isinstance(var_obj, Variable)

    def resolve(self, context, ignore_failures=False):
        if self.is_var:
            try:
                obj = self.var.resolve(context)
            except VariableDoesNotExist:
                if ignore_failures:
                    obj = None
                else:
                    string_if_invalid = context.template.engine.string_if_invalid
                    if string_if_invalid:
                        if '%s' in string_if_invalid:
                            return string_if_invalid % self.var
                        else:
                            return string_if_invalid
                    else:
                        obj = string_if_invalid
        else:
            obj = self.var
        for func, args in self.filters:
            arg_vals = []
            for lookup, arg in args:
                if not lookup:
                    arg_vals.append(mark_safe(arg))
                else:
                    arg_vals.append(arg.resolve(context))
            if getattr(func, 'expects_localtime', False):
                obj = template_localtime(obj, context.use_tz)
            if getattr(func, 'needs_autoescape', False):
                new_obj = func(obj, *arg_vals, autoescape=context.autoescape)
            else:
                new_obj = func(obj, *arg_vals)
            if getattr(func, 'is_safe', False) and isinstance(obj, SafeData):
                obj = mark_safe(new_obj)
            else:
                obj = new_obj
        return obj

    def args_check(name, func, provided):
        provided = list(provided)
        plen = len(provided) + 1
        func = inspect.unwrap(func)
        args, _, _, defaults, _, _, _ = inspect.getfullargspec(func)
        alen = len(args)
        dlen = len(defaults or [])
        if plen < alen - dlen or plen > alen:
            raise TemplateSyntaxError('%s requires %d arguments, %d provided' % (name, alen - dlen, plen))
        return True
    args_check = staticmethod(args_check)

    def __str__(self):
        return self.token

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__qualname__, self.token)