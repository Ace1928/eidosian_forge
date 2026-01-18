from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberun_command
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Retire(kuberun_command.KubeRunCommand, base.DeleteCommand):
    """Retires a KubeRun application in this directory."""
    detailed_help = _DETAILED_HELP
    flags = [flags.EnvironmentFlag()]

    @classmethod
    def Args(cls, parser):
        super(Retire, cls).Args(parser)
        base.DeleteCommand._Flags(parser)
        base.URI_FLAG.RemoveFromParser(parser)

    def Command(self):
        return ['applications', 'retire']