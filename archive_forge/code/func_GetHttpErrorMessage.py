from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import functools
import json
import re
from apitools.base.py import base_api
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.api_lib.functions.v1 import operations
from googlecloudsdk.api_lib.functions.v2 import util as v2_util
from googlecloudsdk.api_lib.storage import storage_api as gcs_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as base_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.generated_clients.apis.cloudfunctions.v1 import cloudfunctions_v1_messages
import six.moves.http_client
def GetHttpErrorMessage(error):
    """Returns a human readable string representation from the http response.

  Args:
    error: HttpException representing the error response.

  Returns:
    A human readable string representation of the error.
  """
    status = error.response.status
    code = error.response.reason
    message = ''
    try:
        data = json.loads(error.content)
        if 'error' in data:
            error_info = data['error']
            if 'message' in error_info:
                message = error_info['message']
            violations = _GetViolationsFromError(error)
            if violations:
                message += '\nProblems:\n' + violations
            if status == 403:
                permission_issues = _GetPermissionErrorDetails(error_info)
                if permission_issues:
                    message += '\nPermission Details:\n' + permission_issues
    except (ValueError, TypeError):
        message = error.content
    return 'ResponseError: status=[{0}], code=[{1}], message=[{2}]'.format(status, code, encoding.Decode(message))