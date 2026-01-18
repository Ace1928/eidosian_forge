from inspect import Arguments
from itertools import chain, tee
from mimetypes import guess_type, add_type
from os.path import splitext
import logging
import operator
import sys
import types
from webob import (Request as WebObRequest, Response as WebObResponse, exc,
from webob.multidict import NestedMultiDict
from .compat import urlparse, izip, is_bound_method as ismethod
from .jsonify import encode as dumps
from .secure import handle_security
from .templating import RendererFactory
from .routing import lookup_controller, NonCanonicalPath
from .util import _cfg, getargspec
from .middleware.recursive import ForwardRequestException
class Pecan(PecanBase):
    """
    Pecan application object. Generally created using ``pecan.make_app``,
    rather than being created manually.

    Creates a Pecan application instance, which is a WSGI application.

    :param root: A string representing a root controller object (e.g.,
                "myapp.controller.root.RootController")
    :param default_renderer: The default template rendering engine to use.
                             Defaults to mako.
    :param template_path: A relative file system path (from the project root)
                          where template files live.  Defaults to 'templates'.
    :param hooks: A callable which returns a list of
                  :class:`pecan.hooks.PecanHook`
    :param custom_renderers: Custom renderer objects, as a dictionary keyed
                             by engine name.
    :param extra_template_vars: Any variables to inject into the template
                                namespace automatically.
    :param force_canonical: A boolean indicating if this project should
                            require canonical URLs.
    :param guess_content_type_from_ext: A boolean indicating if this project
                            should use the extension in the URL for guessing
                            the content type to return.
    :param use_context_locals: When `True`, `pecan.request` and
                               `pecan.response` will be available as
                               thread-local references.
    :param request_cls: Can be used to specify a custom `pecan.request` object.
                        Defaults to `pecan.Request`.
    :param response_cls: Can be used to specify a custom `pecan.response`
                         object.  Defaults to `pecan.Response`.
    """

    def __new__(cls, *args, **kw):
        if kw.get('use_context_locals') is False:
            self = super(Pecan, cls).__new__(ExplicitPecan, *args, **kw)
            self.__init__(*args, **kw)
            return self
        return super(Pecan, cls).__new__(cls)

    def __init__(self, *args, **kw):
        self.init_context_local(kw.get('context_local_factory'))
        super(Pecan, self).__init__(*args, **kw)

    def __call__(self, environ, start_response):
        try:
            state.hooks = []
            state.app = self
            state.controller = None
            state.arguments = None
            return super(Pecan, self).__call__(environ, start_response)
        finally:
            del state.hooks
            del state.request
            del state.response
            del state.controller
            del state.arguments
            del state.app

    def init_context_local(self, local_factory):
        global state
        if local_factory is None:
            from threading import local as local_factory
        state = local_factory()

    def find_controller(self, _state):
        state.request = _state.request
        state.response = _state.response
        controller, args, kw = super(Pecan, self).find_controller(_state)
        state.controller = controller
        state.arguments = _state.arguments
        return (controller, args, kw)

    def handle_hooks(self, hooks, *args, **kw):
        state.hooks = hooks
        return super(Pecan, self).handle_hooks(hooks, *args, **kw)