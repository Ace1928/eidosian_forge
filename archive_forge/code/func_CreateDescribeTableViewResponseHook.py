from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import properties
def CreateDescribeTableViewResponseHook(response, args):
    """Create DescribeTableView from GetVulnerabilityReports response.

  Args:
    response: Response from GetVulnerabilityReports
    args: gcloud invocation args

  Returns:
    DescribeTableView
  """
    del args
    vulnerabilities = encoding.MessageToDict(response.vulnerabilities)
    report_information = {'name': response.name, 'updateTime': response.updateTime}
    return DescribeTableView(vulnerabilities, report_information)