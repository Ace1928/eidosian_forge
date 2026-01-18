from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import batch
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.compute import operation_quota_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
def BatchCheck(self, http_response, exception):
    """Callback for apitools batch responses.

    This will use self.prompted_service_tokens to cache service tokens that
    have already been prompted. In this way, if the same service has multiple
    batch requests and is enabled on the first, the user won't get a bunch of
    superflous messages. Note that this cannot be reused between batch uses
    because of the mutation.

    Args:
      http_response: Deserialized http_wrapper.Response object.
      exception: apiclient.errors.HttpError object if an error occurred.
    """
    if exception is None or not properties.VALUES.core.should_prompt_to_enable_api.GetBool():
        return
    enablement_info = apis.GetApiEnablementInfo(exception)
    if not enablement_info:
        return
    project, service_token, exception = enablement_info
    if service_token not in self.prompted_service_tokens:
        self.prompted_service_tokens.add(service_token)
        apis.PromptToEnableApi(project, service_token, exception, is_batch_request=True)