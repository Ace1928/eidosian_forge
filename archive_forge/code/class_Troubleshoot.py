from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.os_config import troubleshooter
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Troubleshoot(base.Command):
    """Troubleshoot VM Manager issues."""

    def _ResolveInstance(self, holder, compute_client, args):
        """Resolves the arguments into an instance.

    Args:
      holder: the api holder
      compute_client: the compute client
      args: The command line arguments.

    Returns:
      An instance reference to a VM.
    """
        resources = holder.resources
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, resources, scope_lister=flags.GetInstanceZoneScopeLister(compute_client))
        return instance_ref

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        compute_client = holder.client
        instance_ref = self._ResolveInstance(holder, compute_client, args)
        troubleshooter.Troubleshoot(compute_client, instance_ref, self.ReleaseTrack())
        return