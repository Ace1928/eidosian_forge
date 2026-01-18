from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
def DisableXpnAssociatedProject(self, host_project, associated_project):
    """Disassociate the given project from the given XPN host project.

    Args:
      host_project: str, ID of the XPN host project
      associated_project: ID of the project to disassociate from host_project
    """
    xpn_types = self.messages.XpnResourceId.TypeValueValuesEnum
    self._DisableXpnAssociatedResource(host_project, associated_project, xpn_resource_type=xpn_types.PROJECT)