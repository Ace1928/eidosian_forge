from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
class BulkAPIOperationPoller(waiter.CloudOperationPoller):
    """A Poller used by the Bulk API.
  Polls ACM Operations endpoint then calls LIST instead of GET.

  Attributes:
    policy_number: The Access Policy ID that the Poller needs in order to call
      LIST.
  """

    def __init__(self, result_service, operation_service, operation_ref):
        super(BulkAPIOperationPoller, self).__init__(result_service, operation_service)
        policy_id = re.search('^accessPolicies/\\d+', operation_ref.Name())
        if policy_id:
            self.policy_number = policy_id.group()
        else:
            raise ParseResponseError('Could not determine Access Policy ID from operation response.')

    def GetResult(self, operation):
        del operation
        request_type = self.result_service.GetRequestType('List')
        return self.result_service.List(request_type(parent=self.policy_number))