import os
import re
import six
from six.moves import urllib
from routes import request_config
class URLGenerator(object):
    """The URL Generator generates URL's

    It is automatically instantiated by the RoutesMiddleware and put
    into the ``wsgiorg.routing_args`` tuple accessible as::

        url = environ['wsgiorg.routing_args'][0][0]

    Or via the ``routes.url`` key::

        url = environ['routes.url']

    The url object may be instantiated outside of a web context for use
    in testing, however sub_domain support and fully qualified URL's
    cannot be generated without supplying a dict that must contain the
    key ``HTTP_HOST``.

    """

    def __init__(self, mapper, environ):
        """Instantiate the URLGenerator

        ``mapper``
            The mapper object to use when generating routes.
        ``environ``
            The environment dict used in WSGI, alternately, any dict
            that contains at least an ``HTTP_HOST`` value.

        """
        self.mapper = mapper
        if 'SCRIPT_NAME' not in environ:
            environ['SCRIPT_NAME'] = ''
        self.environ = environ

    def __call__(self, *args, **kargs):
        """Generates a URL

        All keys given to url_for are sent to the Routes Mapper instance for
        generation except for::

            anchor          specified the anchor name to be appened to the path
            host            overrides the default (current) host if provided
            protocol        overrides the default (current) protocol if provided
            qualified       creates the URL with the host/port information as
                            needed

        """
        anchor = kargs.get('anchor')
        host = kargs.get('host')
        protocol = kargs.pop('protocol', None)
        qualified = kargs.pop('qualified', None)
        for key in ['anchor', 'host']:
            if kargs.get(key):
                del kargs[key]
            if key + '_' in kargs:
                kargs[key] = kargs.pop(key + '_')
        if 'protocol_' in kargs:
            kargs['protocol_'] = protocol
        route = None
        use_current = '_use_current' in kargs and kargs.pop('_use_current')
        static = False
        encoding = self.mapper.encoding
        url = ''
        more_args = len(args) > 0
        if more_args:
            route = self.mapper._routenames.get(args[0])
        if not route and more_args:
            static = True
            url = args[0]
            if url.startswith('/') and self.environ.get('SCRIPT_NAME'):
                url = self.environ.get('SCRIPT_NAME') + url
            if static:
                if kargs:
                    url += '?'
                    query_args = []
                    for key, val in six.iteritems(kargs):
                        if isinstance(val, (list, tuple)):
                            for value in val:
                                query_args.append('%s=%s' % (urllib.parse.quote(six.text_type(key).encode(encoding)), urllib.parse.quote(six.text_type(value).encode(encoding))))
                        else:
                            query_args.append('%s=%s' % (urllib.parse.quote(six.text_type(key).encode(encoding)), urllib.parse.quote(six.text_type(val).encode(encoding))))
                    url += '&'.join(query_args)
        if not static:
            route_args = []
            if route:
                if self.mapper.hardcode_names:
                    route_args.append(route)
                newargs = route.defaults.copy()
                newargs.update(kargs)
                if route.filter:
                    newargs = route.filter(newargs)
                if not route.static or (route.static and (not route.external)):
                    sub = newargs.get('sub_domain', None)
                    newargs = _subdomain_check(newargs, self.mapper, self.environ)
                    if 'sub_domain' in route.defaults:
                        newargs['sub_domain'] = sub
            elif use_current:
                newargs = _screenargs(kargs, self.mapper, self.environ, force_explicit=True)
            elif 'sub_domain' in kargs:
                newargs = _subdomain_check(kargs, self.mapper, self.environ)
            else:
                newargs = kargs
            anchor = anchor or newargs.pop('_anchor', None)
            host = host or newargs.pop('_host', None)
            if protocol is None:
                protocol = newargs.pop('_protocol', None)
            newargs['_environ'] = self.environ
            url = self.mapper.generate(*route_args, **newargs)
        if anchor is not None:
            url += '#' + _url_quote(anchor, encoding)
        if host or protocol is not None or qualified:
            if 'routes.cached_hostinfo' not in self.environ:
                cache_hostinfo(self.environ)
            hostinfo = self.environ['routes.cached_hostinfo']
            if not host and (not qualified):
                host = hostinfo['host'].split(':')[0]
            elif not host:
                host = hostinfo['host']
            if protocol is None:
                protocol = hostinfo['protocol']
            if protocol != '':
                protocol += ':'
            if url is not None:
                if host[-1] != '/':
                    host += '/'
                url = protocol + '//' + host + url.lstrip('/')
        if not ascii_characters(url) and url is not None:
            raise GenerationException('Can only return a string, got unicode instead: %s' % url)
        if url is None:
            raise GenerationException('Could not generate URL. Called with args: %s %s' % (args, kargs))
        return url

    def current(self, *args, **kwargs):
        """Generate a route that includes params used on the current
        request

        The arguments for this method are identical to ``__call__``
        except that arguments set to None will remove existing route
        matches of the same name from the set of arguments used to
        construct a URL.
        """
        return self(*args, _use_current=True, **kwargs)