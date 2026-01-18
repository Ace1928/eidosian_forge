from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import kuberun_command
from googlecloudsdk.command_lib.kuberun import traffic_pair
from googlecloudsdk.command_lib.kuberun import traffic_printer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.resource import resource_printer
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class UpdateTraffic(kuberun_command.KubeRunCommand):
    """Updates the traffic settings of a KubeRun service."""
    detailed_help = _DETAILED_HELP
    flags = [flags.ClusterConnectionFlags(), flags.NamespaceFlag(), flags.TrafficFlags(), flags.AsyncFlag()]

    @classmethod
    def Args(cls, parser):
        super(UpdateTraffic, cls).Args(parser)
        parser.add_argument('service', help='KubeRun service for which to update the traffic settings.')
        resource_printer.RegisterFormatter(traffic_printer.TRAFFIC_PRINTER_FORMAT, traffic_printer.TrafficPrinter, hidden=True)
        parser.display_info.AddFormat(traffic_printer.TRAFFIC_PRINTER_FORMAT)

    def BuildKubeRunArgs(self, args):
        return [args.service] + super(UpdateTraffic, self).BuildKubeRunArgs(args)

    def Command(self):
        return ['core', 'services', 'update-traffic']

    def SuccessResult(self, out, args):
        if out:
            svc = json.loads(out)
            return traffic_pair.GetTrafficTargetPairsDict(svc)
        else:
            raise exceptions.Error('Failed to update traffic for service [{}]'.format(args.service))