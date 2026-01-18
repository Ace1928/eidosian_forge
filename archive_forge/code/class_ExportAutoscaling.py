from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.core.util import files
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ExportAutoscaling(base.Command):
    """Export autoscaling parameters of a managed instance group to JSON."""

    @staticmethod
    def Args(parser):
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)
        parser.add_argument('--autoscaling-file', metavar='PATH', required=True, help='Path of the file to which autoscaling configuration will be written.')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        igm_ref = instance_groups_flags.CreateGroupReference(client, holder.resources, args)
        autoscaler = managed_instance_groups_utils.AutoscalerForMigByRef(client, holder.resources, igm_ref)
        if autoscaler:
            autoscaler_dict = encoding.MessageToDict(autoscaler)
            for f in _IGNORED_FIELDS:
                if f in autoscaler_dict:
                    del autoscaler_dict[f]
        else:
            autoscaler_dict = None
        files.WriteFileContents(args.autoscaling_file, json.dumps(autoscaler_dict))