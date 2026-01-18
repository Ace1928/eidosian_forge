from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddIPAliasRelatedFlags(parser, autopilot=False):
    """Adds flags related to IP aliases to the parser.

  Args:
    parser: A given parser.
    autopilot: True if the cluster is in autopilot mode.
  """
    if autopilot:
        must_specify_enable_ip_alias = ''
        must_specify_enable_ip_alias_new_block = ''
    else:
        must_specify_enable_ip_alias = "\nCannot be specified unless '--enable-ip-alias' option is also specified.\n"
        must_specify_enable_ip_alias_new_block = "\n\nCannot be specified unless '--enable-ip-alias' option is also specified.\n"
    parser.add_argument('--services-ipv4-cidr', metavar='CIDR', help="Set the IP range for the services IPs.\n\nCan be specified as a netmask size (e.g. '/20') or as in CIDR notion\n(e.g. '10.100.0.0/20'). If given as a netmask size, the IP range will\nbe chosen automatically from the available space in the network.\n\nIf unspecified, the services CIDR range will be chosen with a default\nmask size.{}\n".format(must_specify_enable_ip_alias_new_block))
    parser.add_argument('--create-subnetwork', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='Create a new subnetwork for the cluster. The name and range of the\nsubnetwork can be customized via optional \'name\' and \'range\' key-value\npairs.\n\n\'name\' specifies the name of the subnetwork to be created.\n\n\'range\' specifies the IP range for the new subnetwork. This can either\nbe a netmask size (e.g. \'/20\') or a CIDR range (e.g. \'10.0.0.0/20\').\nIf a netmask size is specified, the IP is automatically taken from the\nfree space in the cluster\'s network.\n\nExamples:\n\nCreate a new subnetwork with a default name and size.\n\n  $ {{command}} --create-subnetwork ""\n\nCreate a new subnetwork named "my-subnet" with netmask of size 21.\n\n  $ {{command}} --create-subnetwork name=my-subnet,range=/21\n\nCreate a new subnetwork with a default name with the primary range of\n10.100.0.0/16.\n\n  $ {{command}} --create-subnetwork range=10.100.0.0/16\n\nCreate a new subnetwork with the name "my-subnet" with a default range.\n\n  $ {{command}} --create-subnetwork name=my-subnet\n\n{}Cannot be used in conjunction with \'--subnetwork\' option.\n'.format(must_specify_enable_ip_alias))
    parser.add_argument('--cluster-secondary-range-name', metavar='NAME', help="Set the secondary range to be used as the source for pod IPs. Alias\nranges will be allocated from this secondary range.  NAME must be the\nname of an existing secondary range in the cluster subnetwork.\n\n{}Cannot be used with '--create-subnetwork' option.\n".format(must_specify_enable_ip_alias))
    parser.add_argument('--services-secondary-range-name', metavar='NAME', help="Set the secondary range to be used for services (e.g. ClusterIPs).\nNAME must be the name of an existing secondary range in the cluster\nsubnetwork.\n\n{}Cannot be used with '--create-subnetwork' option.\n".format(must_specify_enable_ip_alias))