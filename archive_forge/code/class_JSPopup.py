import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
class JSPopup(URLResource):
    """
    >>> u = URL('/')
    >>> u = u / 'view'
    >>> j = u.js_popup(content='view')
    >>> j.html
    '<a href="/view" onclick="window.open(&#x27;/view&#x27;, &#x27;_blank&#x27;); return false" target="_blank">view</a>'
    """
    default_params = {'tag': 'a', 'target': '_blank'}

    def _add_vars(self, vars):
        button = self
        for var in ('width', 'height', 'stripped', 'content'):
            if var in vars:
                button = button.param(**{var: vars.pop(var)})
        return button.var(**vars)

    def _window_args(self):
        p = self.params
        features = []
        if p.get('stripped'):
            p['location'] = p['status'] = p['toolbar'] = '0'
        for param in 'channelmode directories fullscreen location menubar resizable scrollbars status titlebar'.split():
            if param not in p:
                continue
            v = p[param]
            if v not in ('yes', 'no', '1', '0'):
                if v:
                    v = '1'
                else:
                    v = '0'
            features.append('%s=%s' % (param, v))
        for param in 'height left top width':
            if not p.get(param):
                continue
            features.append('%s=%s' % (param, p[param]))
        args = [self.href, p['target']]
        if features:
            args.append(','.join(features))
        return ', '.join(map(js_repr, args))

    def _html_attrs(self):
        attrs = list(self.attrs.items())
        onclick = 'window.open(%s); return false' % self._window_args()
        attrs.insert(0, ('target', self.params['target']))
        attrs.insert(0, ('onclick', onclick))
        attrs.insert(0, ('href', self.href))
        return attrs

    def _get_content(self):
        if not self.params.get('content'):
            raise ValueError('You must give a content param to %r generate anchor tags' % self)
        return self.params['content']

    def _add_positional(self, args):
        return self.addpath(*args)