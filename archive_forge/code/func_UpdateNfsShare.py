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
def UpdateNfsShare(self, nfs_share_resource, labels, allowed_clients):
    """Update an existing nfs share resource."""
    updated_fields = []
    updated_allowed_clients = []
    if labels is not None:
        updated_fields.append('labels')
    if allowed_clients is not None:
        updated_fields.append('allowedClients')
        updated_allowed_clients = allowed_clients
    nfs_share_msg = self.messages.NfsShare(name=nfs_share_resource.RelativeName(), labels=labels, allowedClients=updated_allowed_clients)
    request = self.messages.BaremetalsolutionProjectsLocationsNfsSharesPatchRequest(name=nfs_share_resource.RelativeName(), nfsShare=nfs_share_msg, updateMask=','.join(updated_fields))
    return self.nfs_shares_service.Patch(request)