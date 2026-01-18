from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import util
from googlecloudsdk.command_lib.functions.v1.remove_iam_policy_binding import command as command_v1
from googlecloudsdk.command_lib.functions.v2.remove_iam_policy_binding import command as command_v2
from googlecloudsdk.command_lib.iam import iam_util
def _RunV2(self, args):
    return command_v2.Run(args, self.ReleaseTrack())