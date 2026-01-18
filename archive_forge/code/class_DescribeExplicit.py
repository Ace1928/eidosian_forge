from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc.settings import flags
from googlecloudsdk.command_lib.scc.settings import utils
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DescribeExplicit(base.DescribeCommand):
    """Display module settings of Security Command Center(SCC)."""
    detailed_help = {'DESCRIPTION': '      Describe explicit module settings of Security Command Center(SCC).\n      ', 'EXAMPLES': '        To describe the explict \'OPEN_FIREWALL\' module setting in service \'SECURITY_HEALTH_ANALYTICS\' of project "12345", run:\n\n          $ {command} --project=12345 --service=SECURITY_HEALTH_ANALYTICS --module=OPEN_FIREWALL\n      '}

    @staticmethod
    def Args(parser):
        flags.ExtractRequiredFlags(parser)
        flags.AddServiceArgument(parser)
        flags.AddModuleArgument(parser)

    def Run(self, args):
        """Call corresponding APIs based on the flag."""
        response = utils.SettingsClient().DescribeServiceExplicit(args)
        configs = response.modules.additionalProperties if response.modules else []
        state = [p.value.moduleEnablementState for p in configs if p.key == args.module]
        if state:
            return state[0]
        else:
            log.status.Print('No setting found for module {}. The module may not exist or no explicit setting is set.'.format(args.module))
            return None