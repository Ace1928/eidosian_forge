import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
class CookieJar:
    """Collection of HTTP cookies.

    You may not need to know about this class: try
    urllib.request.build_opener(HTTPCookieProcessor).open(url).
    """
    non_word_re = re.compile('\\W')
    quote_re = re.compile('([\\"\\\\])')
    strict_domain_re = re.compile('\\.?[^.]*')
    domain_re = re.compile('[^.]*')
    dots_re = re.compile('^\\.+')
    magic_re = re.compile('^\\#LWP-Cookies-(\\d+\\.\\d+)', re.ASCII)

    def __init__(self, policy=None):
        if policy is None:
            policy = DefaultCookiePolicy()
        self._policy = policy
        self._cookies_lock = _threading.RLock()
        self._cookies = {}

    def set_policy(self, policy):
        self._policy = policy

    def _cookies_for_domain(self, domain, request):
        cookies = []
        if not self._policy.domain_return_ok(domain, request):
            return []
        _debug('Checking %s for cookies to return', domain)
        cookies_by_path = self._cookies[domain]
        for path in cookies_by_path.keys():
            if not self._policy.path_return_ok(path, request):
                continue
            cookies_by_name = cookies_by_path[path]
            for cookie in cookies_by_name.values():
                if not self._policy.return_ok(cookie, request):
                    _debug('   not returning cookie')
                    continue
                _debug("   it's a match")
                cookies.append(cookie)
        return cookies

    def _cookies_for_request(self, request):
        """Return a list of cookies to be returned to server."""
        cookies = []
        for domain in self._cookies.keys():
            cookies.extend(self._cookies_for_domain(domain, request))
        return cookies

    def _cookie_attrs(self, cookies):
        """Return a list of cookie-attributes to be returned to server.

        like ['foo="bar"; $Path="/"', ...]

        The $Version attribute is also added when appropriate (currently only
        once per request).

        """
        cookies.sort(key=lambda a: len(a.path), reverse=True)
        version_set = False
        attrs = []
        for cookie in cookies:
            version = cookie.version
            if not version_set:
                version_set = True
                if version > 0:
                    attrs.append('$Version=%s' % version)
            if cookie.value is not None and self.non_word_re.search(cookie.value) and (version > 0):
                value = self.quote_re.sub('\\\\\\1', cookie.value)
            else:
                value = cookie.value
            if cookie.value is None:
                attrs.append(cookie.name)
            else:
                attrs.append('%s=%s' % (cookie.name, value))
            if version > 0:
                if cookie.path_specified:
                    attrs.append('$Path="%s"' % cookie.path)
                if cookie.domain.startswith('.'):
                    domain = cookie.domain
                    if not cookie.domain_initial_dot and domain.startswith('.'):
                        domain = domain[1:]
                    attrs.append('$Domain="%s"' % domain)
                if cookie.port is not None:
                    p = '$Port'
                    if cookie.port_specified:
                        p = p + '="%s"' % cookie.port
                    attrs.append(p)
        return attrs

    def add_cookie_header(self, request):
        """Add correct Cookie: header to request (urllib.request.Request object).

        The Cookie2 header is also added unless policy.hide_cookie2 is true.

        """
        _debug('add_cookie_header')
        self._cookies_lock.acquire()
        try:
            self._policy._now = self._now = int(time.time())
            cookies = self._cookies_for_request(request)
            attrs = self._cookie_attrs(cookies)
            if attrs:
                if not request.has_header('Cookie'):
                    request.add_unredirected_header('Cookie', '; '.join(attrs))
            if self._policy.rfc2965 and (not self._policy.hide_cookie2) and (not request.has_header('Cookie2')):
                for cookie in cookies:
                    if cookie.version != 1:
                        request.add_unredirected_header('Cookie2', '$Version="1"')
                        break
        finally:
            self._cookies_lock.release()
        self.clear_expired_cookies()

    def _normalized_cookie_tuples(self, attrs_set):
        """Return list of tuples containing normalised cookie information.

        attrs_set is the list of lists of key,value pairs extracted from
        the Set-Cookie or Set-Cookie2 headers.

        Tuples are name, value, standard, rest, where name and value are the
        cookie name and value, standard is a dictionary containing the standard
        cookie-attributes (discard, secure, version, expires or max-age,
        domain, path and port) and rest is a dictionary containing the rest of
        the cookie-attributes.

        """
        cookie_tuples = []
        boolean_attrs = ('discard', 'secure')
        value_attrs = ('version', 'expires', 'max-age', 'domain', 'path', 'port', 'comment', 'commenturl')
        for cookie_attrs in attrs_set:
            name, value = cookie_attrs[0]
            max_age_set = False
            bad_cookie = False
            standard = {}
            rest = {}
            for k, v in cookie_attrs[1:]:
                lc = k.lower()
                if lc in value_attrs or lc in boolean_attrs:
                    k = lc
                if k in boolean_attrs and v is None:
                    v = True
                if k in standard:
                    continue
                if k == 'domain':
                    if v is None:
                        _debug('   missing value for domain attribute')
                        bad_cookie = True
                        break
                    v = v.lower()
                if k == 'expires':
                    if max_age_set:
                        continue
                    if v is None:
                        _debug('   missing or invalid value for expires attribute: treating as session cookie')
                        continue
                if k == 'max-age':
                    max_age_set = True
                    try:
                        v = int(v)
                    except ValueError:
                        _debug('   missing or invalid (non-numeric) value for max-age attribute')
                        bad_cookie = True
                        break
                    k = 'expires'
                    v = self._now + v
                if k in value_attrs or k in boolean_attrs:
                    if v is None and k not in ('port', 'comment', 'commenturl'):
                        _debug('   missing value for %s attribute' % k)
                        bad_cookie = True
                        break
                    standard[k] = v
                else:
                    rest[k] = v
            if bad_cookie:
                continue
            cookie_tuples.append((name, value, standard, rest))
        return cookie_tuples

    def _cookie_from_cookie_tuple(self, tup, request):
        name, value, standard, rest = tup
        domain = standard.get('domain', Absent)
        path = standard.get('path', Absent)
        port = standard.get('port', Absent)
        expires = standard.get('expires', Absent)
        version = standard.get('version', None)
        if version is not None:
            try:
                version = int(version)
            except ValueError:
                return None
        secure = standard.get('secure', False)
        discard = standard.get('discard', False)
        comment = standard.get('comment', None)
        comment_url = standard.get('commenturl', None)
        if path is not Absent and path != '':
            path_specified = True
            path = escape_path(path)
        else:
            path_specified = False
            path = request_path(request)
            i = path.rfind('/')
            if i != -1:
                if version == 0:
                    path = path[:i]
                else:
                    path = path[:i + 1]
            if len(path) == 0:
                path = '/'
        domain_specified = domain is not Absent
        domain_initial_dot = False
        if domain_specified:
            domain_initial_dot = bool(domain.startswith('.'))
        if domain is Absent:
            req_host, erhn = eff_request_host(request)
            domain = erhn
        elif not domain.startswith('.'):
            domain = '.' + domain
        port_specified = False
        if port is not Absent:
            if port is None:
                port = request_port(request)
            else:
                port_specified = True
                port = re.sub('\\s+', '', port)
        else:
            port = None
        if expires is Absent:
            expires = None
            discard = True
        elif expires <= self._now:
            try:
                self.clear(domain, path, name)
            except KeyError:
                pass
            _debug("Expiring cookie, domain='%s', path='%s', name='%s'", domain, path, name)
            return None
        return Cookie(version, name, value, port, port_specified, domain, domain_specified, domain_initial_dot, path, path_specified, secure, expires, discard, comment, comment_url, rest)

    def _cookies_from_attrs_set(self, attrs_set, request):
        cookie_tuples = self._normalized_cookie_tuples(attrs_set)
        cookies = []
        for tup in cookie_tuples:
            cookie = self._cookie_from_cookie_tuple(tup, request)
            if cookie:
                cookies.append(cookie)
        return cookies

    def _process_rfc2109_cookies(self, cookies):
        rfc2109_as_ns = getattr(self._policy, 'rfc2109_as_netscape', None)
        if rfc2109_as_ns is None:
            rfc2109_as_ns = not self._policy.rfc2965
        for cookie in cookies:
            if cookie.version == 1:
                cookie.rfc2109 = True
                if rfc2109_as_ns:
                    cookie.version = 0

    def make_cookies(self, response, request):
        """Return sequence of Cookie objects extracted from response object."""
        headers = response.info()
        rfc2965_hdrs = headers.get_all('Set-Cookie2', [])
        ns_hdrs = headers.get_all('Set-Cookie', [])
        self._policy._now = self._now = int(time.time())
        rfc2965 = self._policy.rfc2965
        netscape = self._policy.netscape
        if not rfc2965_hdrs and (not ns_hdrs) or (not ns_hdrs and (not rfc2965)) or (not rfc2965_hdrs and (not netscape)) or (not netscape and (not rfc2965)):
            return []
        try:
            cookies = self._cookies_from_attrs_set(split_header_words(rfc2965_hdrs), request)
        except Exception:
            _warn_unhandled_exception()
            cookies = []
        if ns_hdrs and netscape:
            try:
                ns_cookies = self._cookies_from_attrs_set(parse_ns_headers(ns_hdrs), request)
            except Exception:
                _warn_unhandled_exception()
                ns_cookies = []
            self._process_rfc2109_cookies(ns_cookies)
            if rfc2965:
                lookup = {}
                for cookie in cookies:
                    lookup[cookie.domain, cookie.path, cookie.name] = None

                def no_matching_rfc2965(ns_cookie, lookup=lookup):
                    key = (ns_cookie.domain, ns_cookie.path, ns_cookie.name)
                    return key not in lookup
                ns_cookies = filter(no_matching_rfc2965, ns_cookies)
            if ns_cookies:
                cookies.extend(ns_cookies)
        return cookies

    def set_cookie_if_ok(self, cookie, request):
        """Set a cookie if policy says it's OK to do so."""
        self._cookies_lock.acquire()
        try:
            self._policy._now = self._now = int(time.time())
            if self._policy.set_ok(cookie, request):
                self.set_cookie(cookie)
        finally:
            self._cookies_lock.release()

    def set_cookie(self, cookie):
        """Set a cookie, without checking whether or not it should be set."""
        c = self._cookies
        self._cookies_lock.acquire()
        try:
            if cookie.domain not in c:
                c[cookie.domain] = {}
            c2 = c[cookie.domain]
            if cookie.path not in c2:
                c2[cookie.path] = {}
            c3 = c2[cookie.path]
            c3[cookie.name] = cookie
        finally:
            self._cookies_lock.release()

    def extract_cookies(self, response, request):
        """Extract cookies from response, where allowable given the request."""
        _debug('extract_cookies: %s', response.info())
        self._cookies_lock.acquire()
        try:
            for cookie in self.make_cookies(response, request):
                if self._policy.set_ok(cookie, request):
                    _debug(' setting cookie: %s', cookie)
                    self.set_cookie(cookie)
        finally:
            self._cookies_lock.release()

    def clear(self, domain=None, path=None, name=None):
        """Clear some cookies.

        Invoking this method without arguments will clear all cookies.  If
        given a single argument, only cookies belonging to that domain will be
        removed.  If given two arguments, cookies belonging to the specified
        path within that domain are removed.  If given three arguments, then
        the cookie with the specified name, path and domain is removed.

        Raises KeyError if no matching cookie exists.

        """
        if name is not None:
            if domain is None or path is None:
                raise ValueError('domain and path must be given to remove a cookie by name')
            del self._cookies[domain][path][name]
        elif path is not None:
            if domain is None:
                raise ValueError('domain must be given to remove cookies by path')
            del self._cookies[domain][path]
        elif domain is not None:
            del self._cookies[domain]
        else:
            self._cookies = {}

    def clear_session_cookies(self):
        """Discard all session cookies.

        Note that the .save() method won't save session cookies anyway, unless
        you ask otherwise by passing a true ignore_discard argument.

        """
        self._cookies_lock.acquire()
        try:
            for cookie in self:
                if cookie.discard:
                    self.clear(cookie.domain, cookie.path, cookie.name)
        finally:
            self._cookies_lock.release()

    def clear_expired_cookies(self):
        """Discard all expired cookies.

        You probably don't need to call this method: expired cookies are never
        sent back to the server (provided you're using DefaultCookiePolicy),
        this method is called by CookieJar itself every so often, and the
        .save() method won't save expired cookies anyway (unless you ask
        otherwise by passing a true ignore_expires argument).

        """
        self._cookies_lock.acquire()
        try:
            now = time.time()
            for cookie in self:
                if cookie.is_expired(now):
                    self.clear(cookie.domain, cookie.path, cookie.name)
        finally:
            self._cookies_lock.release()

    def __iter__(self):
        return deepvalues(self._cookies)

    def __len__(self):
        """Return number of contained cookies."""
        i = 0
        for cookie in self:
            i = i + 1
        return i

    def __repr__(self):
        r = []
        for cookie in self:
            r.append(repr(cookie))
        return '<%s[%s]>' % (self.__class__.__name__, ', '.join(r))

    def __str__(self):
        r = []
        for cookie in self:
            r.append(str(cookie))
        return '<%s[%s]>' % (self.__class__.__name__, ', '.join(r))