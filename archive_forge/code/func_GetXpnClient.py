from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
def GetXpnClient(release_track):
    holder = base_classes.ComputeApiHolder(release_track)
    return XpnClient(holder.client)