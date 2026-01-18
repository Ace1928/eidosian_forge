from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.filestore import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddNetworkArg(parser):
    """Adds a --network flag to the given parser.

  Args:
    parser: argparse parser.
  """
    network_arg_spec = {'name': str, 'reserved-ip-range': str, 'connect-mode': str}
    network_help = "        Network configuration for a Cloud Filestore instance. Specifying\n        `reserved-ip-range` and `connect-mode` is optional.\n        *name*::: The name of the Google Compute Engine\n        [VPC network](/compute/docs/networks-and-firewalls#networks) to which\n        the instance is connected.\n        *reserved-ip-range*::: The `reserved-ip-range` can have one of the\n        following two types of values: a CIDR range value when using\n        DIRECT_PEERING connect mode or an allocated IP address range\n        (https://cloud.google.com/compute/docs/ip-addresses/reserve-static-internal-ip-address)\n        when using PRIVATE_SERVICE_ACCESS connect mode. When the name of an\n        allocated IP address range is specified, it must be one of the ranges\n        associated with the private service access connection. When specified as\n        a direct CIDR value, it must be a /29 CIDR block for Basic tier or a /24\n        CIDR block for High Scale, Zonal, Enterprise or Regional tier in one of the internal IP\n        address ranges (https://www.arin.net/knowledge/address_filters.html)\n        that identifies the range of IP addresses reserved for this instance.\n        For example, 10.0.0.0/29 or 192.168.0.0/24. The range you specify can't\n        overlap with either existing subnets or assigned IP address ranges for\n        other Cloud Filestore instances in the selected VPC network.\n        *connect-mode*::: Network connection mode used by instances.\n        CONNECT_MODE must be one of: DIRECT_PEERING or PRIVATE_SERVICE_ACCESS.\n  "
    parser.add_argument('--network', type=arg_parsers.ArgDict(spec=network_arg_spec, required_keys=['name']), required=True, help=network_help)