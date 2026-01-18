from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import retry
def _ShouldRetryHttpError(exc_type, exc_value, exc_traceback, state):
    """Whether to retry the request when receiving errors.

  Args:
    exc_type: type of the raised exception.
    exc_value: the instance of the raise the exception.
    exc_traceback: Traceback, traceback encapsulating the call stack at the the
      point where the exception occurred.
    state: RetryerState, state of the retryer.

  Returns:
    True if exception and is due to NOT_FOUND or INVALID_ARGUMENT.
  """
    del exc_value, exc_traceback, state
    return exc_type == apitools_exceptions.HttpBadRequestError or exc_type == apitools_exceptions.HttpNotFoundError