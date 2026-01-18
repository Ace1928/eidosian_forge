from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import ssh as tpu_ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ScpAlpha(Scp):
    """Copy files to and from a Cloud TPU VM via SCP (Alpha)."""
    _ENABLE_IAP = True
    _ENABLE_BATCHING = True