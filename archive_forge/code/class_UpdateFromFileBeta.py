from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class UpdateFromFileBeta(UpdateFromFile):
    """Update a Compute Engine virtual machine instance using a configuration file."""
    _support_secure_tag = False