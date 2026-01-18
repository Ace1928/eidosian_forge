from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os.path
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import ssh as tpu_ssh_utils
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SshAlpha(Ssh):
    """SSH into a Cloud TPU VM (Alpha)."""
    _ENABLE_IAP = True
    _ENABLE_BATCHING = True