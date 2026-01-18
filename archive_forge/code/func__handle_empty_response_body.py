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
def _handle_empty_response_body(self, state):
    if state.response.status_int == 200:
        if isinstance(state.response.app_iter, types.GeneratorType):
            a, b = tee(state.response.app_iter)
            try:
                next(a)
            except StopIteration:
                state.response.status = 204
            finally:
                state.response.app_iter = b
        else:
            text = None
            if state.response.charset:
                try:
                    text = state.response.text
                except UnicodeDecodeError:
                    pass
            if not any((state.response.body, text)):
                state.response.status = 204
    if state.response.status_int in (204, 304):
        state.response.content_type = None