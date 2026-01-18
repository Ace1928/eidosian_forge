import functools
import inspect
import re
import string
from importlib import import_module
from pickle import PicklingError
from urllib.parse import quote
from asgiref.local import Local
from django.conf import settings
from django.core.checks import Error, Warning
from django.core.checks.urls import check_resolver
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
from django.utils.regex_helper import _lazy_re_compile, normalize
from django.utils.translation import get_language
from .converters import get_converter
from .exceptions import NoReverseMatch, Resolver404
from .utils import get_callable
def _reverse_with_prefix(self, lookup_view, _prefix, *args, **kwargs):
    if args and kwargs:
        raise ValueError("Don't mix *args and **kwargs in call to reverse()!")
    if not self._populated:
        self._populate()
    possibilities = self.reverse_dict.getlist(lookup_view)
    for possibility, pattern, defaults, converters in possibilities:
        for result, params in possibility:
            if args:
                if len(args) != len(params):
                    continue
                candidate_subs = dict(zip(params, args))
            else:
                if set(kwargs).symmetric_difference(params).difference(defaults):
                    continue
                matches = True
                for k, v in defaults.items():
                    if k in params:
                        continue
                    if kwargs.get(k, v) != v:
                        matches = False
                        break
                if not matches:
                    continue
                candidate_subs = kwargs
            text_candidate_subs = {}
            match = True
            for k, v in candidate_subs.items():
                if k in converters:
                    try:
                        text_candidate_subs[k] = converters[k].to_url(v)
                    except ValueError:
                        match = False
                        break
                else:
                    text_candidate_subs[k] = str(v)
            if not match:
                continue
            candidate_pat = _prefix.replace('%', '%%') + result
            if re.search('^%s%s' % (re.escape(_prefix), pattern), candidate_pat % text_candidate_subs):
                url = quote(candidate_pat % text_candidate_subs, safe=RFC3986_SUBDELIMS + '/~:@')
                return escape_leading_slashes(url)
    m = getattr(lookup_view, '__module__', None)
    n = getattr(lookup_view, '__name__', None)
    if m is not None and n is not None:
        lookup_view_s = '%s.%s' % (m, n)
    else:
        lookup_view_s = lookup_view
    patterns = [pattern for _, pattern, _, _ in possibilities]
    if patterns:
        if args:
            arg_msg = "arguments '%s'" % (args,)
        elif kwargs:
            arg_msg = "keyword arguments '%s'" % kwargs
        else:
            arg_msg = 'no arguments'
        msg = "Reverse for '%s' with %s not found. %d pattern(s) tried: %s" % (lookup_view_s, arg_msg, len(patterns), patterns)
    else:
        msg = "Reverse for '%(view)s' not found. '%(view)s' is not a valid view function or pattern name." % {'view': lookup_view_s}
    raise NoReverseMatch(msg)