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
def DeleteStoragePool(self, storagepool_ref, async_):
    """Deletes an existing Cloud NetApp Storage Pool."""
    request = self.messages.NetappProjectsLocationsStoragePoolsDeleteRequest(name=storagepool_ref.RelativeName())
    return self._DeleteStoragePool(async_, request)