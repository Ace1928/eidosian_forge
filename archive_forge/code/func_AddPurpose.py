from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddPurpose(parser, support_psc_google_apis):
    """Adds purpose flag."""
    choices = ['VPC_PEERING', 'SHARED_LOADBALANCER_VIP', 'GCE_ENDPOINT', 'IPSEC_INTERCONNECT']
    if support_psc_google_apis:
        choices.append('PRIVATE_SERVICE_CONNECT')
    parser.add_argument('--purpose', choices=choices, type=lambda x: x.upper(), help='      The purpose of the address resource. This field is not applicable to\n      external addresses.\n      ')