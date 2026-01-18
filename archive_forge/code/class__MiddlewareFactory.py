from webob.compat import (
from webob.request import Request
from webob.exc import HTTPException
class _MiddlewareFactory(object):
    """A middleware that has not yet been bound to an application or
    configured.
    """

    def __init__(self, wrapper_class, middleware, kw):
        self.wrapper_class = wrapper_class
        self.middleware = middleware
        self.kw = kw

    def __repr__(self):
        return '<%s at %s wrapping %r>' % (self.__class__.__name__, id(self), self.middleware)

    def __call__(self, app=None, **config):
        kw = self.kw.copy()
        kw.update(config)
        return self.wrapper_class.middleware(self.middleware, app, **kw)