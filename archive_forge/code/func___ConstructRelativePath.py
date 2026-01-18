import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
def __ConstructRelativePath(self, method_config, request, relative_path=None):
    """Determine the relative path for request."""
    python_param_names = util.MapParamNames(method_config.path_params, type(request))
    params = dict([(param, getattr(request, param, None)) for param in python_param_names])
    params = util.MapRequestParams(params, type(request))
    return util.ExpandRelativePath(method_config, params, relative_path=relative_path)