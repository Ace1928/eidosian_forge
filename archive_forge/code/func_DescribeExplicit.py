from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.scc.settings import exceptions as scc_exceptions
from googlecloudsdk.core import properties
def DescribeExplicit(self, args):
    """Describe settings of organization."""
    path = GenerateParent(args) + 'securityCenterSettings'
    try:
        request_message = self.message_module.SecuritycenterOrganizationsGetSecurityCenterSettingsRequest(name=path)
        return self.service_client.organizations.GetSecurityCenterSettings(request_message)
    except exceptions.HttpNotFoundError:
        raise scc_exceptions.SecurityCenterSettingsException('Invalid argument {}'.format(path))