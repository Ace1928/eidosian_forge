from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.service_directory import services
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.service_directory import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class RemoveIamPolicyBindingBeta(RemoveIamPolicyBinding):
    """Removes IAM policy binding from a service."""

    def GetReleaseTrack(self):
        return base.ReleaseTrack.BETA