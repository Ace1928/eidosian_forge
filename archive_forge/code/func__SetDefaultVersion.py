from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _SetDefaultVersion(new_version, api_client):
    """Sets the given version as the default.

  Args:
    new_version: Version, The version to promote.
    api_client: appengine_api_client.AppengineApiClient to use to make requests.
  """
    metrics.CustomTimedEvent(metric_names.SET_DEFAULT_VERSION_API_START)

    def ShouldRetry(exc_type, unused_exc_value, unused_traceback, unused_state):
        return issubclass(exc_type, apitools_exceptions.HttpError)
    try:
        retryer = retry.Retryer(max_retrials=3, exponential_sleep_multiplier=2)
        retryer.RetryOnException(api_client.SetDefaultVersion, [new_version.service, new_version.id], should_retry_if=ShouldRetry, sleep_ms=1000)
    except retry.MaxRetrialsException as e:
        unused_result, exc_info = e.last_result
        if exc_info:
            exceptions.reraise(exc_info[1], tb=exc_info[2])
        else:
            raise exceptions.InternalError()
    metrics.CustomTimedEvent(metric_names.SET_DEFAULT_VERSION_API)