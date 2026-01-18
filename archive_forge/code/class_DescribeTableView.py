from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import properties
class DescribeTableView:
    """View model for vulnerability-reports describe."""

    def __init__(self, vulnerabilities, report_information):
        self.vulnerabilities = vulnerabilities
        self.report_information = report_information