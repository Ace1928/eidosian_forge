from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.machine_images import flags as machine_image_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ImportBeta(Import):
    """Import a machine image into Compute Engine from OVF for Beta."""

    @classmethod
    def Args(cls, parser):
        super(ImportBeta, cls).Args(parser)
        daisy_utils.AddExtraCommonDaisyArgs(parser)