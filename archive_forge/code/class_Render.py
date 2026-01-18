from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberun_command
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Render(kuberun_command.KubeRunCommand, base.ExportCommand):
    """Render KubeRun application to generate the yaml resource configuration."""
    detailed_help = _DETAILED_HELP
    flags = [_OutFlag(), flags.EnvironmentFlag()]

    def Command(self):
        return ['render']