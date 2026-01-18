from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.health_checks import flags as health_checks_flags
def AddAutohealingArgs(autohealing_params_group):
    """Adds autohealing-related commandline arguments to parser."""
    autohealing_params_group.add_argument('--initial-delay', type=_InitialDelayValidator, help="      Specifies the number of seconds that a new VM takes to initialize and run\n      its startup script. During a VM's initial delay period, the MIG ignores\n      unsuccessful health checks because the VM might be in the startup process.\n      This prevents the MIG from prematurely recreating a VM. If the health\n      check receives a healthy response during the initial delay, it indicates\n      that the startup process is complete and the VM is ready. The value of\n      initial delay must be between 0 and 3600 seconds. The default value is 0.\n      See $ gcloud topic datetimes for information on duration formats.\n      ")
    health_check_group = autohealing_params_group.add_mutually_exclusive_group()
    health_check_group.add_argument('--http-health-check', help='HTTP health check object used for autohealing instances in this group.', action=actions.DeprecationAction('http-health-check', warn='HttpHealthCheck is deprecated. Use --health-check instead.'))
    health_check_group.add_argument('--https-health-check', help='HTTPS health check object used for autohealing instances in this group.', action=actions.DeprecationAction('https-health-check', warn='HttpsHealthCheck is deprecated. Use --health-check instead.'))
    HEALTH_CHECK_ARG.AddArgument(health_check_group)