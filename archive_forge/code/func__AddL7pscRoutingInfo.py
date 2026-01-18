from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddL7pscRoutingInfo(parser):
    """Adds l7psc routing info arguments for PSC network endpoint groups."""
    psc_target_service_help = '      PSC target service name to add to the private service connect network\n      endpoint groups (NEG).\n  '
    parser.add_argument('--psc-target-service', help=psc_target_service_help)