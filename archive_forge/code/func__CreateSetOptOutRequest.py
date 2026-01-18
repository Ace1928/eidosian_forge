from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import e2e_integrity
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
def _CreateSetOptOutRequest(self, args):
    messages = cloudkms_base.GetMessagesAlphaModule()
    req = messages.CloudkmsProjectsSetProjectOptOutStateRequest(name=args.project, setProjectOptOutStateRequest=messages.SetProjectOptOutStateRequest(value=not args.undo))
    return req