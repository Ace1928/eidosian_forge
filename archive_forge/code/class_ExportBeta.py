from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ExportBeta(Export):
    """Export a Compute Engine image for Beta release track."""

    @classmethod
    def Args(cls, parser):
        super(ExportBeta, cls).Args(parser)
        daisy_utils.AddExtraCommonDaisyArgs(parser)

    def _RunImageExport(self, args, export_args, tags, output_filter):
        return daisy_utils.RunImageExport(args, export_args, tags, _OUTPUT_FILTER, release_track=self.ReleaseTrack().id.lower() if self.ReleaseTrack() else None, docker_image_tag=args.docker_image_tag)