from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def UpdateActiveDirectory(self, activedirectory_ref, activedirectory_config, update_mask):
    """Send a Patch request for the Cloud NetApp Active Directory."""
    update_request = self.messages.NetappProjectsLocationsActiveDirectoriesPatchRequest(activeDirectory=activedirectory_config, name=activedirectory_ref.RelativeName(), updateMask=update_mask)
    update_op = self.client.projects_locations_activeDirectories.Patch(update_request)
    return update_op