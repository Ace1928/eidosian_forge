from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.service_directory import locations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.service_directory import resource_args
def GetReleaseTrack(self):
    return base.ReleaseTrack.BETA