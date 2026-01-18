from __future__ import annotations
import typing as t
import warnings
from pprint import pformat
from threading import Lock
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlunsplit
from .._internal import _get_environ
from .._internal import _wsgi_decoding_dance
from ..datastructures import ImmutableDict
from ..datastructures import MultiDict
from ..exceptions import BadHost
from ..exceptions import HTTPException
from ..exceptions import MethodNotAllowed
from ..exceptions import NotFound
from ..urls import _urlencode
from ..wsgi import get_host
from .converters import DEFAULT_CONVERTERS
from .exceptions import BuildError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .exceptions import RequestRedirect
from .exceptions import WebsocketMismatch
from .matcher import StateMachineMatcher
from .rules import _simple_rule_re
from .rules import Rule
class MapAdapter:
    """Returned by :meth:`Map.bind` or :meth:`Map.bind_to_environ` and does
    the URL matching and building based on runtime information.
    """

    def __init__(self, map: Map, server_name: str, script_name: str, subdomain: str | None, url_scheme: str, path_info: str, default_method: str, query_args: t.Mapping[str, t.Any] | str | None=None):
        self.map = map
        self.server_name = server_name
        if not script_name.endswith('/'):
            script_name += '/'
        self.script_name = script_name
        self.subdomain = subdomain
        self.url_scheme = url_scheme
        self.path_info = path_info
        self.default_method = default_method
        self.query_args = query_args
        self.websocket = self.url_scheme in {'ws', 'wss'}

    def dispatch(self, view_func: t.Callable[[str, t.Mapping[str, t.Any]], WSGIApplication], path_info: str | None=None, method: str | None=None, catch_http_exceptions: bool=False) -> WSGIApplication:
        """Does the complete dispatching process.  `view_func` is called with
        the endpoint and a dict with the values for the view.  It should
        look up the view function, call it, and return a response object
        or WSGI application.  http exceptions are not caught by default
        so that applications can display nicer error messages by just
        catching them by hand.  If you want to stick with the default
        error messages you can pass it ``catch_http_exceptions=True`` and
        it will catch the http exceptions.

        Here a small example for the dispatch usage::

            from werkzeug.wrappers import Request, Response
            from werkzeug.wsgi import responder
            from werkzeug.routing import Map, Rule

            def on_index(request):
                return Response('Hello from the index')

            url_map = Map([Rule('/', endpoint='index')])
            views = {'index': on_index}

            @responder
            def application(environ, start_response):
                request = Request(environ)
                urls = url_map.bind_to_environ(environ)
                return urls.dispatch(lambda e, v: views[e](request, **v),
                                     catch_http_exceptions=True)

        Keep in mind that this method might return exception objects, too, so
        use :class:`Response.force_type` to get a response object.

        :param view_func: a function that is called with the endpoint as
                          first argument and the value dict as second.  Has
                          to dispatch to the actual view function with this
                          information.  (see above)
        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        :param catch_http_exceptions: set to `True` to catch any of the
                                      werkzeug :class:`HTTPException`\\s.
        """
        try:
            try:
                endpoint, args = self.match(path_info, method)
            except RequestRedirect as e:
                return e
            return view_func(endpoint, args)
        except HTTPException as e:
            if catch_http_exceptions:
                return e
            raise

    @t.overload
    def match(self, path_info: str | None=None, method: str | None=None, return_rule: t.Literal[False]=False, query_args: t.Mapping[str, t.Any] | str | None=None, websocket: bool | None=None) -> tuple[str, t.Mapping[str, t.Any]]:
        ...

    @t.overload
    def match(self, path_info: str | None=None, method: str | None=None, return_rule: t.Literal[True]=True, query_args: t.Mapping[str, t.Any] | str | None=None, websocket: bool | None=None) -> tuple[Rule, t.Mapping[str, t.Any]]:
        ...

    def match(self, path_info: str | None=None, method: str | None=None, return_rule: bool=False, query_args: t.Mapping[str, t.Any] | str | None=None, websocket: bool | None=None) -> tuple[str | Rule, t.Mapping[str, t.Any]]:
        """The usage is simple: you just pass the match method the current
        path info as well as the method (which defaults to `GET`).  The
        following things can then happen:

        - you receive a `NotFound` exception that indicates that no URL is
          matching.  A `NotFound` exception is also a WSGI application you
          can call to get a default page not found page (happens to be the
          same object as `werkzeug.exceptions.NotFound`)

        - you receive a `MethodNotAllowed` exception that indicates that there
          is a match for this URL but not for the current request method.
          This is useful for RESTful applications.

        - you receive a `RequestRedirect` exception with a `new_url`
          attribute.  This exception is used to notify you about a request
          Werkzeug requests from your WSGI application.  This is for example the
          case if you request ``/foo`` although the correct URL is ``/foo/``
          You can use the `RequestRedirect` instance as response-like object
          similar to all other subclasses of `HTTPException`.

        - you receive a ``WebsocketMismatch`` exception if the only
          match is a WebSocket rule but the bind is an HTTP request, or
          if the match is an HTTP rule but the bind is a WebSocket
          request.

        - you get a tuple in the form ``(endpoint, arguments)`` if there is
          a match (unless `return_rule` is True, in which case you get a tuple
          in the form ``(rule, arguments)``)

        If the path info is not passed to the match method the default path
        info of the map is used (defaults to the root URL if not defined
        explicitly).

        All of the exceptions raised are subclasses of `HTTPException` so they
        can be used as WSGI responses. They will all render generic error or
        redirect pages.

        Here is a small example for matching:

        >>> m = Map([
        ...     Rule('/', endpoint='index'),
        ...     Rule('/downloads/', endpoint='downloads/index'),
        ...     Rule('/downloads/<int:id>', endpoint='downloads/show')
        ... ])
        >>> urls = m.bind("example.com", "/")
        >>> urls.match("/", "GET")
        ('index', {})
        >>> urls.match("/downloads/42")
        ('downloads/show', {'id': 42})

        And here is what happens on redirect and missing URLs:

        >>> urls.match("/downloads")
        Traceback (most recent call last):
          ...
        RequestRedirect: http://example.com/downloads/
        >>> urls.match("/missing")
        Traceback (most recent call last):
          ...
        NotFound: 404 Not Found

        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        :param return_rule: return the rule that matched instead of just the
                            endpoint (defaults to `False`).
        :param query_args: optional query arguments that are used for
                           automatic redirects as string or dictionary.  It's
                           currently not possible to use the query arguments
                           for URL matching.
        :param websocket: Match WebSocket instead of HTTP requests. A
            websocket request has a ``ws`` or ``wss``
            :attr:`url_scheme`. This overrides that detection.

        .. versionadded:: 1.0
            Added ``websocket``.

        .. versionchanged:: 0.8
            ``query_args`` can be a string.

        .. versionadded:: 0.7
            Added ``query_args``.

        .. versionadded:: 0.6
            Added ``return_rule``.
        """
        self.map.update()
        if path_info is None:
            path_info = self.path_info
        if query_args is None:
            query_args = self.query_args or {}
        method = (method or self.default_method).upper()
        if websocket is None:
            websocket = self.websocket
        domain_part = self.server_name
        if not self.map.host_matching and self.subdomain is not None:
            domain_part = self.subdomain
        path_part = f'/{path_info.lstrip('/')}' if path_info else ''
        try:
            result = self.map._matcher.match(domain_part, path_part, method, websocket)
        except RequestPath as e:
            new_path = quote(e.path_info, safe="!$&'()*+,/:;=@")
            raise RequestRedirect(self.make_redirect_url(new_path, query_args)) from None
        except RequestAliasRedirect as e:
            raise RequestRedirect(self.make_alias_redirect_url(f'{domain_part}|{path_part}', e.endpoint, e.matched_values, method, query_args)) from None
        except NoMatch as e:
            if e.have_match_for:
                raise MethodNotAllowed(valid_methods=list(e.have_match_for)) from None
            if e.websocket_mismatch:
                raise WebsocketMismatch() from None
            raise NotFound() from None
        else:
            rule, rv = result
            if self.map.redirect_defaults:
                redirect_url = self.get_default_redirect(rule, method, rv, query_args)
                if redirect_url is not None:
                    raise RequestRedirect(redirect_url)
            if rule.redirect_to is not None:
                if isinstance(rule.redirect_to, str):

                    def _handle_match(match: t.Match[str]) -> str:
                        value = rv[match.group(1)]
                        return rule._converters[match.group(1)].to_url(value)
                    redirect_url = _simple_rule_re.sub(_handle_match, rule.redirect_to)
                else:
                    redirect_url = rule.redirect_to(self, **rv)
                if self.subdomain:
                    netloc = f'{self.subdomain}.{self.server_name}'
                else:
                    netloc = self.server_name
                raise RequestRedirect(urljoin(f'{self.url_scheme or 'http'}://{netloc}{self.script_name}', redirect_url))
            if return_rule:
                return (rule, rv)
            else:
                return (rule.endpoint, rv)

    def test(self, path_info: str | None=None, method: str | None=None) -> bool:
        """Test if a rule would match.  Works like `match` but returns `True`
        if the URL matches, or `False` if it does not exist.

        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        """
        try:
            self.match(path_info, method)
        except RequestRedirect:
            pass
        except HTTPException:
            return False
        return True

    def allowed_methods(self, path_info: str | None=None) -> t.Iterable[str]:
        """Returns the valid methods that match for a given path.

        .. versionadded:: 0.7
        """
        try:
            self.match(path_info, method='--')
        except MethodNotAllowed as e:
            return e.valid_methods
        except HTTPException:
            pass
        return []

    def get_host(self, domain_part: str | None) -> str:
        """Figures out the full host name for the given domain part.  The
        domain part is a subdomain in case host matching is disabled or
        a full host name.
        """
        if self.map.host_matching:
            if domain_part is None:
                return self.server_name
            return domain_part
        if domain_part is None:
            subdomain = self.subdomain
        else:
            subdomain = domain_part
        if subdomain:
            return f'{subdomain}.{self.server_name}'
        else:
            return self.server_name

    def get_default_redirect(self, rule: Rule, method: str, values: t.MutableMapping[str, t.Any], query_args: t.Mapping[str, t.Any] | str) -> str | None:
        """A helper that returns the URL to redirect to if it finds one.
        This is used for default redirecting only.

        :internal:
        """
        assert self.map.redirect_defaults
        for r in self.map._rules_by_endpoint[rule.endpoint]:
            if r is rule:
                break
            if r.provides_defaults_for(rule) and r.suitable_for(values, method):
                values.update(r.defaults)
                domain_part, path = r.build(values)
                return self.make_redirect_url(path, query_args, domain_part=domain_part)
        return None

    def encode_query_args(self, query_args: t.Mapping[str, t.Any] | str) -> str:
        if not isinstance(query_args, str):
            return _urlencode(query_args)
        return query_args

    def make_redirect_url(self, path_info: str, query_args: t.Mapping[str, t.Any] | str | None=None, domain_part: str | None=None) -> str:
        """Creates a redirect URL.

        :internal:
        """
        if query_args is None:
            query_args = self.query_args
        if query_args:
            query_str = self.encode_query_args(query_args)
        else:
            query_str = None
        scheme = self.url_scheme or 'http'
        host = self.get_host(domain_part)
        path = '/'.join((self.script_name.strip('/'), path_info.lstrip('/')))
        return urlunsplit((scheme, host, path, query_str, None))

    def make_alias_redirect_url(self, path: str, endpoint: str, values: t.Mapping[str, t.Any], method: str, query_args: t.Mapping[str, t.Any] | str) -> str:
        """Internally called to make an alias redirect URL."""
        url = self.build(endpoint, values, method, append_unknown=False, force_external=True)
        if query_args:
            url += f'?{self.encode_query_args(query_args)}'
        assert url != path, 'detected invalid alias setting. No canonical URL found'
        return url

    def _partial_build(self, endpoint: str, values: t.Mapping[str, t.Any], method: str | None, append_unknown: bool) -> tuple[str, str, bool] | None:
        """Helper for :meth:`build`.  Returns subdomain and path for the
        rule that accepts this endpoint, values and method.

        :internal:
        """
        if method is None:
            rv = self._partial_build(endpoint, values, self.default_method, append_unknown)
            if rv is not None:
                return rv
        first_match = None
        for rule in self.map._rules_by_endpoint.get(endpoint, ()):
            if rule.suitable_for(values, method):
                build_rv = rule.build(values, append_unknown)
                if build_rv is not None:
                    rv = (build_rv[0], build_rv[1], rule.websocket)
                    if self.map.host_matching:
                        if rv[0] == self.server_name:
                            return rv
                        elif first_match is None:
                            first_match = rv
                    else:
                        return rv
        return first_match

    def build(self, endpoint: str, values: t.Mapping[str, t.Any] | None=None, method: str | None=None, force_external: bool=False, append_unknown: bool=True, url_scheme: str | None=None) -> str:
        """Building URLs works pretty much the other way round.  Instead of
        `match` you call `build` and pass it the endpoint and a dict of
        arguments for the placeholders.

        The `build` function also accepts an argument called `force_external`
        which, if you set it to `True` will force external URLs. Per default
        external URLs (include the server name) will only be used if the
        target URL is on a different subdomain.

        >>> m = Map([
        ...     Rule('/', endpoint='index'),
        ...     Rule('/downloads/', endpoint='downloads/index'),
        ...     Rule('/downloads/<int:id>', endpoint='downloads/show')
        ... ])
        >>> urls = m.bind("example.com", "/")
        >>> urls.build("index", {})
        '/'
        >>> urls.build("downloads/show", {'id': 42})
        '/downloads/42'
        >>> urls.build("downloads/show", {'id': 42}, force_external=True)
        'http://example.com/downloads/42'

        Because URLs cannot contain non ASCII data you will always get
        bytes back.  Non ASCII characters are urlencoded with the
        charset defined on the map instance.

        Additional values are converted to strings and appended to the URL as
        URL querystring parameters:

        >>> urls.build("index", {'q': 'My Searchstring'})
        '/?q=My+Searchstring'

        When processing those additional values, lists are furthermore
        interpreted as multiple values (as per
        :py:class:`werkzeug.datastructures.MultiDict`):

        >>> urls.build("index", {'q': ['a', 'b', 'c']})
        '/?q=a&q=b&q=c'

        Passing a ``MultiDict`` will also add multiple values:

        >>> urls.build("index", MultiDict((('p', 'z'), ('q', 'a'), ('q', 'b'))))
        '/?p=z&q=a&q=b'

        If a rule does not exist when building a `BuildError` exception is
        raised.

        The build method accepts an argument called `method` which allows you
        to specify the method you want to have an URL built for if you have
        different methods for the same endpoint specified.

        :param endpoint: the endpoint of the URL to build.
        :param values: the values for the URL to build.  Unhandled values are
                       appended to the URL as query parameters.
        :param method: the HTTP method for the rule if there are different
                       URLs for different methods on the same endpoint.
        :param force_external: enforce full canonical external URLs. If the URL
                               scheme is not provided, this will generate
                               a protocol-relative URL.
        :param append_unknown: unknown parameters are appended to the generated
                               URL as query string argument.  Disable this
                               if you want the builder to ignore those.
        :param url_scheme: Scheme to use in place of the bound
            :attr:`url_scheme`.

        .. versionchanged:: 2.0
            Added the ``url_scheme`` parameter.

        .. versionadded:: 0.6
           Added the ``append_unknown`` parameter.
        """
        self.map.update()
        if values:
            if isinstance(values, MultiDict):
                values = {k: v[0] if len(v) == 1 else v for k, v in dict.items(values) if len(v) != 0}
            else:
                values = {k: v for k, v in values.items() if v is not None}
        else:
            values = {}
        rv = self._partial_build(endpoint, values, method, append_unknown)
        if rv is None:
            raise BuildError(endpoint, values, method, self)
        domain_part, path, websocket = rv
        host = self.get_host(domain_part)
        if url_scheme is None:
            url_scheme = self.url_scheme
        secure = url_scheme in {'https', 'wss'}
        if websocket:
            force_external = True
            url_scheme = 'wss' if secure else 'ws'
        elif url_scheme:
            url_scheme = 'https' if secure else 'http'
        if not force_external and (self.map.host_matching and host == self.server_name or (not self.map.host_matching and domain_part == self.subdomain)):
            return f'{self.script_name.rstrip('/')}/{path.lstrip('/')}'
        scheme = f'{url_scheme}:' if url_scheme else ''
        return f'{scheme}//{host}{self.script_name[:-1]}/{path.lstrip('/')}'