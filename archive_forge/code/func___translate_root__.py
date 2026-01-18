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
def __translate_root__(self, item):
    """
        Creates a root controller instance from a string root, e.g.,

        > __translate_root__("myproject.controllers.RootController")
        myproject.controllers.RootController()

        :param item: The string to the item
        """
    if '.' in item:
        parts = item.split('.')
        name = '.'.join(parts[:-1])
        fromlist = parts[-1:]
        module = __import__(name, fromlist=fromlist)
        kallable = getattr(module, parts[-1])
        msg = '%s does not represent a callable class or function.'
        if not callable(kallable):
            raise TypeError(msg % item)
        return kallable()
    raise ImportError('No item named %s' % item)