from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
def CatchHTTPErrorRaiseHTTPExceptionDecorator(run_func):

    def Wrapper(*args, **kwargs):
        try:
            return run_func(*args, **kwargs)
        except apitools_exceptions.HttpError as error:
            exc = HttpException(error, format_str)
            core_exceptions.reraise(exc)
    return Wrapper