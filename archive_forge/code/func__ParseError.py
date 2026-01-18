from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.command_lib.bms import util
import six
def _ParseError(error):
    """Returns a best-effort error message created from an API client error."""
    if isinstance(error, apitools_exceptions.HttpError):
        parsed_error = apilib_exceptions.HttpException(error, error_format='{message}')
        error_message = parsed_error.message
    else:
        error_message = six.text_type(error)
    return error_message