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
class ExplicitPecan(PecanBase):

    def get_args(self, state, all_params, remainder, argspec, im_self):
        if hasattr(state.controller, '__self__'):
            _repr = '.'.join((state.controller.__self__.__class__.__module__, state.controller.__self__.__class__.__name__, state.controller.__name__))
        else:
            _repr = '.'.join((state.controller.__module__, state.controller.__name__))
        signature_error = TypeError('When `use_context_locals` is `False`, pecan passes an explicit reference to the request and response as the first two arguments to the controller.\nChange the `%s` signature to accept exactly 2 initial arguments (req, resp)' % _repr)
        try:
            positional = argspec.args[:]
            positional.pop(1)
            positional.pop(1)
            argspec = argspec._replace(args=positional)
        except IndexError:
            raise signature_error
        args, varargs, kwargs = super(ExplicitPecan, self).get_args(state, all_params, remainder, argspec, im_self)
        if ismethod(state.controller):
            args = [state.request, state.response] + args
        else:
            args[1:1] = [state.request, state.response]
        return (args, varargs, kwargs)