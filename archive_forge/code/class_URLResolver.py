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
class URLResolver:

    def __init__(self, pattern, urlconf_name, default_kwargs=None, app_name=None, namespace=None):
        self.pattern = pattern
        self.urlconf_name = urlconf_name
        self.callback = None
        self.default_kwargs = default_kwargs or {}
        self.namespace = namespace
        self.app_name = app_name
        self._reverse_dict = {}
        self._namespace_dict = {}
        self._app_dict = {}
        self._callback_strs = set()
        self._populated = False
        self._local = Local()

    def __repr__(self):
        if isinstance(self.urlconf_name, list) and self.urlconf_name:
            urlconf_repr = '<%s list>' % self.urlconf_name[0].__class__.__name__
        else:
            urlconf_repr = repr(self.urlconf_name)
        return '<%s %s (%s:%s) %s>' % (self.__class__.__name__, urlconf_repr, self.app_name, self.namespace, self.pattern.describe())

    def check(self):
        messages = []
        for pattern in self.url_patterns:
            messages.extend(check_resolver(pattern))
        messages.extend(self._check_custom_error_handlers())
        return messages or self.pattern.check()

    def _check_custom_error_handlers(self):
        messages = []
        for status_code, num_parameters in [(400, 2), (403, 2), (404, 2), (500, 1)]:
            try:
                handler = self.resolve_error_handler(status_code)
            except (ImportError, ViewDoesNotExist) as e:
                path = getattr(self.urlconf_module, 'handler%s' % status_code)
                msg = "The custom handler{status_code} view '{path}' could not be imported.".format(status_code=status_code, path=path)
                messages.append(Error(msg, hint=str(e), id='urls.E008'))
                continue
            signature = inspect.signature(handler)
            args = [None] * num_parameters
            try:
                signature.bind(*args)
            except TypeError:
                msg = "The custom handler{status_code} view '{path}' does not take the correct number of arguments ({args}).".format(status_code=status_code, path=handler.__module__ + '.' + handler.__qualname__, args='request, exception' if num_parameters == 2 else 'request')
                messages.append(Error(msg, id='urls.E007'))
        return messages

    def _populate(self):
        if getattr(self._local, 'populating', False):
            return
        try:
            self._local.populating = True
            lookups = MultiValueDict()
            namespaces = {}
            apps = {}
            language_code = get_language()
            for url_pattern in reversed(self.url_patterns):
                p_pattern = url_pattern.pattern.regex.pattern
                p_pattern = p_pattern.removeprefix('^')
                if isinstance(url_pattern, URLPattern):
                    self._callback_strs.add(url_pattern.lookup_str)
                    bits = normalize(url_pattern.pattern.regex.pattern)
                    lookups.appendlist(url_pattern.callback, (bits, p_pattern, url_pattern.default_args, url_pattern.pattern.converters))
                    if url_pattern.name is not None:
                        lookups.appendlist(url_pattern.name, (bits, p_pattern, url_pattern.default_args, url_pattern.pattern.converters))
                else:
                    url_pattern._populate()
                    if url_pattern.app_name:
                        apps.setdefault(url_pattern.app_name, []).append(url_pattern.namespace)
                        namespaces[url_pattern.namespace] = (p_pattern, url_pattern)
                    else:
                        for name in url_pattern.reverse_dict:
                            for matches, pat, defaults, converters in url_pattern.reverse_dict.getlist(name):
                                new_matches = normalize(p_pattern + pat)
                                lookups.appendlist(name, (new_matches, p_pattern + pat, {**defaults, **url_pattern.default_kwargs}, {**self.pattern.converters, **url_pattern.pattern.converters, **converters}))
                        for namespace, (prefix, sub_pattern) in url_pattern.namespace_dict.items():
                            current_converters = url_pattern.pattern.converters
                            sub_pattern.pattern.converters.update(current_converters)
                            namespaces[namespace] = (p_pattern + prefix, sub_pattern)
                        for app_name, namespace_list in url_pattern.app_dict.items():
                            apps.setdefault(app_name, []).extend(namespace_list)
                    self._callback_strs.update(url_pattern._callback_strs)
            self._namespace_dict[language_code] = namespaces
            self._app_dict[language_code] = apps
            self._reverse_dict[language_code] = lookups
            self._populated = True
        finally:
            self._local.populating = False

    @property
    def reverse_dict(self):
        language_code = get_language()
        if language_code not in self._reverse_dict:
            self._populate()
        return self._reverse_dict[language_code]

    @property
    def namespace_dict(self):
        language_code = get_language()
        if language_code not in self._namespace_dict:
            self._populate()
        return self._namespace_dict[language_code]

    @property
    def app_dict(self):
        language_code = get_language()
        if language_code not in self._app_dict:
            self._populate()
        return self._app_dict[language_code]

    @staticmethod
    def _extend_tried(tried, pattern, sub_tried=None):
        if sub_tried is None:
            tried.append([pattern])
        else:
            tried.extend(([pattern, *t] for t in sub_tried))

    @staticmethod
    def _join_route(route1, route2):
        """Join two routes, without the starting ^ in the second route."""
        if not route1:
            return route2
        route2 = route2.removeprefix('^')
        return route1 + route2

    def _is_callback(self, name):
        if not self._populated:
            self._populate()
        return name in self._callback_strs

    def resolve(self, path):
        path = str(path)
        tried = []
        match = self.pattern.match(path)
        if match:
            new_path, args, kwargs = match
            for pattern in self.url_patterns:
                try:
                    sub_match = pattern.resolve(new_path)
                except Resolver404 as e:
                    self._extend_tried(tried, pattern, e.args[0].get('tried'))
                else:
                    if sub_match:
                        sub_match_dict = {**kwargs, **self.default_kwargs}
                        sub_match_dict.update(sub_match.kwargs)
                        sub_match_args = sub_match.args
                        if not sub_match_dict:
                            sub_match_args = args + sub_match.args
                        current_route = '' if isinstance(pattern, URLPattern) else str(pattern.pattern)
                        self._extend_tried(tried, pattern, sub_match.tried)
                        return ResolverMatch(sub_match.func, sub_match_args, sub_match_dict, sub_match.url_name, [self.app_name] + sub_match.app_names, [self.namespace] + sub_match.namespaces, self._join_route(current_route, sub_match.route), tried, captured_kwargs=sub_match.captured_kwargs, extra_kwargs={**self.default_kwargs, **sub_match.extra_kwargs})
                    tried.append([pattern])
            raise Resolver404({'tried': tried, 'path': new_path})
        raise Resolver404({'path': path})

    @cached_property
    def urlconf_module(self):
        if isinstance(self.urlconf_name, str):
            return import_module(self.urlconf_name)
        else:
            return self.urlconf_name

    @cached_property
    def url_patterns(self):
        patterns = getattr(self.urlconf_module, 'urlpatterns', self.urlconf_module)
        try:
            iter(patterns)
        except TypeError as e:
            msg = "The included URLconf '{name}' does not appear to have any patterns in it. If you see the 'urlpatterns' variable with valid patterns in the file then the issue is probably caused by a circular import."
            raise ImproperlyConfigured(msg.format(name=self.urlconf_name)) from e
        return patterns

    def resolve_error_handler(self, view_type):
        callback = getattr(self.urlconf_module, 'handler%s' % view_type, None)
        if not callback:
            from django.conf import urls
            callback = getattr(urls, 'handler%s' % view_type)
        return get_callable(callback)

    def reverse(self, lookup_view, *args, **kwargs):
        return self._reverse_with_prefix(lookup_view, '', *args, **kwargs)

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