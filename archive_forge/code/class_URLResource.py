import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
class URLResource:
    """
    This is an abstract superclass for different kinds of URLs
    """
    default_params = {}

    def __init__(self, url, vars=None, attrs=None, params=None):
        self.url = url or '/'
        self.vars = vars or []
        self.attrs = attrs or {}
        self.params = self.default_params.copy()
        self.original_params = params or {}
        if params:
            self.params.update(params)

    def from_environ(cls, environ, with_query_string=True, with_path_info=True, script_name=None, path_info=None, querystring=None):
        url = request.construct_url(environ, with_query_string=False, with_path_info=with_path_info, script_name=script_name, path_info=path_info)
        if with_query_string:
            if querystring is None:
                vars = request.parse_querystring(environ)
            else:
                vars = parse_qsl(querystring, keep_blank_values=True, strict_parsing=False)
        else:
            vars = None
        v = cls(url, vars=vars)
        return v
    from_environ = classmethod(from_environ)

    def __call__(self, *args, **kw):
        res = self._add_positional(args)
        res = res._add_vars(kw)
        return res

    def __getitem__(self, item):
        if '=' in item:
            name, value = item.split('=', 1)
            return self._add_vars({unquote(name): unquote(value)})
        return self._add_positional((item,))

    def attr(self, **kw):
        for key in kw.keys():
            if key.endswith('_'):
                kw[key[:-1]] = kw[key]
                del kw[key]
        new_attrs = self.attrs.copy()
        new_attrs.update(kw)
        return self.__class__(self.url, vars=self.vars, attrs=new_attrs, params=self.original_params)

    def param(self, **kw):
        new_params = self.original_params.copy()
        new_params.update(kw)
        return self.__class__(self.url, vars=self.vars, attrs=self.attrs, params=new_params)

    def coerce_vars(self, vars):
        global variabledecode
        need_variable_encode = False
        for key, value in vars.items():
            if isinstance(value, dict):
                need_variable_encode = True
            if key.endswith('_'):
                vars[key[:-1]] = vars[key]
                del vars[key]
        if need_variable_encode:
            if variabledecode is None:
                from formencode import variabledecode
            vars = variabledecode.variable_encode(vars)
        return vars

    def var(self, **kw):
        kw = self.coerce_vars(kw)
        new_vars = self.vars + list(kw.items())
        return self.__class__(self.url, vars=new_vars, attrs=self.attrs, params=self.original_params)

    def setvar(self, **kw):
        """
        Like ``.var(...)``, except overwrites keys, where .var simply
        extends the keys.  Setting a variable to None here will
        effectively delete it.
        """
        kw = self.coerce_vars(kw)
        new_vars = []
        for name, values in self.vars:
            if name in kw:
                continue
            new_vars.append((name, values))
        new_vars.extend(kw.items())
        return self.__class__(self.url, vars=new_vars, attrs=self.attrs, params=self.original_params)

    def setvars(self, **kw):
        """
        Creates a copy of this URL, but with all the variables set/reset
        (like .setvar(), except clears past variables at the same time)
        """
        return self.__class__(self.url, vars=kw.items(), attrs=self.attrs, params=self.original_params)

    def addpath(self, *paths):
        u = self
        for path in paths:
            path = str(path).lstrip('/')
            new_url = u.url
            if not new_url.endswith('/'):
                new_url += '/'
            u = u.__class__(new_url + path, vars=u.vars, attrs=u.attrs, params=u.original_params)
        return u
    __truediv__ = addpath

    def become(self, OtherClass):
        return OtherClass(self.url, vars=self.vars, attrs=self.attrs, params=self.original_params)

    def href__get(self):
        s = self.url
        if self.vars:
            s += '?'
            vars = []
            for name, val in self.vars:
                if isinstance(val, (list, tuple)):
                    val = [v for v in val if v is not None]
                elif val is None:
                    continue
                vars.append((name, val))
            s += urlencode(vars, True)
        return s
    href = property(href__get)

    def __repr__(self):
        base = '<%s %s' % (self.__class__.__name__, self.href or "''")
        if self.attrs:
            base += ' attrs(%s)' % ' '.join(['%s="%s"' % (html_quote(n), html_quote(v)) for n, v in self.attrs.items()])
        if self.original_params:
            base += ' params(%s)' % ', '.join(['%s=%r' % (n, v) for n, v in self.attrs.items()])
        return base + '>'

    def html__get(self):
        if not self.params.get('tag'):
            raise ValueError("You cannot get the HTML of %r until you set the 'tag' param'" % self)
        content = self._get_content()
        tag = '<%s' % self.params.get('tag')
        attrs = ' '.join(['%s="%s"' % (html_quote(n), html_quote(v)) for n, v in self._html_attrs()])
        if attrs:
            tag += ' ' + attrs
        tag += self._html_extra()
        if content is None:
            return tag + ' />'
        else:
            return '%s>%s</%s>' % (tag, content, self.params.get('tag'))
    html = property(html__get)

    def _html_attrs(self):
        return self.attrs.items()

    def _html_extra(self):
        return ''

    def _get_content(self):
        """
        Return the content for a tag (for self.html); return None
        for an empty tag (like ``<img />``)
        """
        raise NotImplementedError

    def _add_vars(self, vars):
        raise NotImplementedError

    def _add_positional(self, args):
        raise NotImplementedError