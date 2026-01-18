import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def NormalizeScopes(scope_spec):
    """Normalize scope_spec to a set of strings."""
    if isinstance(scope_spec, six.string_types):
        scope_spec = six.ensure_str(scope_spec)
        return set(scope_spec.split(' '))
    elif isinstance(scope_spec, Iterable):
        scope_spec = [six.ensure_str(x) for x in scope_spec]
        return set(scope_spec)
    raise exceptions.TypecheckError('NormalizeScopes expected string or iterable, found %s' % (type(scope_spec),))