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
class URLPattern:

    def __init__(self, pattern, callback, default_args=None, name=None):
        self.pattern = pattern
        self.callback = callback
        self.default_args = default_args or {}
        self.name = name

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.pattern.describe())

    def check(self):
        warnings = self._check_pattern_name()
        warnings.extend(self.pattern.check())
        warnings.extend(self._check_callback())
        return warnings

    def _check_pattern_name(self):
        """
        Check that the pattern name does not contain a colon.
        """
        if self.pattern.name is not None and ':' in self.pattern.name:
            warning = Warning("Your URL pattern {} has a name including a ':'. Remove the colon, to avoid ambiguous namespace references.".format(self.pattern.describe()), id='urls.W003')
            return [warning]
        else:
            return []

    def _check_callback(self):
        from django.views import View
        view = self.callback
        if inspect.isclass(view) and issubclass(view, View):
            return [Error('Your URL pattern %s has an invalid view, pass %s.as_view() instead of %s.' % (self.pattern.describe(), view.__name__, view.__name__), id='urls.E009')]
        return []

    def resolve(self, path):
        match = self.pattern.match(path)
        if match:
            new_path, args, captured_kwargs = match
            kwargs = {**captured_kwargs, **self.default_args}
            return ResolverMatch(self.callback, args, kwargs, self.pattern.name, route=str(self.pattern), captured_kwargs=captured_kwargs, extra_kwargs=self.default_args)

    @cached_property
    def lookup_str(self):
        """
        A string that identifies the view (e.g. 'path.to.view_function' or
        'path.to.ClassBasedView').
        """
        callback = self.callback
        if isinstance(callback, functools.partial):
            callback = callback.func
        if hasattr(callback, 'view_class'):
            callback = callback.view_class
        elif not hasattr(callback, '__name__'):
            return callback.__module__ + '.' + callback.__class__.__name__
        return callback.__module__ + '.' + callback.__qualname__