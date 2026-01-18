from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
def AddDestinationGroup(parser):
    destination_group = parser.add_group(mutex=True, required=True)
    AddDestinationOrgArgs(destination_group)
    AddDestinationFolderArgs(destination_group)