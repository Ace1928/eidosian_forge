from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddSubnet(parser):
    """Adds subnet argument for creating network endpoint groups."""
    help_text = '      Name of the subnet to which all network endpoints belong.\n\n      If not specified, network endpoints may belong to any subnetwork in the\n      region where the network endpoint group is created.\n  '
    subnet_applicable_types = ['`gce-vm-ip-port`']
    subnet_applicable_types.append('`gce-vm-ip`')
    subnet_applicable_types.append('`private-service-connect`')
    help_text += '\n      This is only supported for NEGs with endpoint type {0}.\n      For Private Service Connect NEGs, you can optionally specify --network and\n      --subnet if --psc-target-service points to a published service. If\n      --psc-target-service points to the regional service endpoint of a Google\n      API, do not specify --network or --subnet.\n  '.format(_JoinWithOr(subnet_applicable_types))
    parser.add_argument('--subnet', help=help_text)