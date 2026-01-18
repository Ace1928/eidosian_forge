from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.command_lib.bms import util
import six
def RenameNetwork(self, network_resource, new_name):
    """Rename an existing network resource."""
    rename_network_request = self.messages.RenameNetworkRequest(newNetworkId=new_name)
    request = self.messages.BaremetalsolutionProjectsLocationsNetworksRenameRequest(name=network_resource.RelativeName(), renameNetworkRequest=rename_network_request)
    return self.networks_service.Rename(request)