from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetApiEnablementInfo(exception):
    """Returns the API Enablement info or None if prompting is not necessary.

  Args:
    exception (apitools_exceptions.HttpError): Exception if an error occurred.

  Returns:
    tuple[str]: The project, service token, exception tuple to be used for
      prompting to enable the API.

  Raises:
    api_exceptions.HttpException: If gcloud should not prompt to enable the API.
  """
    parsed_error = api_exceptions.HttpException(exception)
    if parsed_error.payload.status_code != API_ENABLEMENT_ERROR_EXPECTED_STATUS_CODE:
        return None
    enablement_info = api_enablement.GetApiEnablementInfo(parsed_error.payload.status_message)
    if enablement_info:
        return enablement_info + (parsed_error,)
    return None