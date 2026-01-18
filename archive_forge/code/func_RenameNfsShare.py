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
def RenameNfsShare(self, nfs_share_resource, new_name):
    """Rename an existing nfs share resource."""
    rename_nfs_share_request = self.messages.RenameNfsShareRequest(newNfsshareId=new_name)
    request = self.messages.BaremetalsolutionProjectsLocationsNfsSharesRenameRequest(name=nfs_share_resource.RelativeName(), renameNfsShareRequest=rename_nfs_share_request)
    return self.nfs_shares_service.Rename(request)