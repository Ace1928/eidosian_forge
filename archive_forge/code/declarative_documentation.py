from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
Create ListTableRow from ListOSPolicyAssignmentReports response.

  Args:
    response: Response from ListOSPolicyAssignmentReports
    args: gcloud args

  Returns:
    ListTableRows of summarized os policy assignment reports
  