from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddDefaultPort(parser):
    """Adds default port argument for creating network endpoint groups."""
    help_text = "    The default port to use if the port number is not specified in the network\n    endpoint.\n\n    If this flag isn't specified for a NEG with endpoint type `gce-vm-ip-port`\n    or `non-gcp-private-ip-port`, then every network endpoint in the network\n    endpoint group must have a port specified. For a global NEG with endpoint\n    type `internet-ip-port` and `internet-fqdn-port` if the default port is not\n    specified, the well-known port for your backend protocol is used (80 for\n    HTTP, 443 for HTTPS).\n\n    This flag is not supported for NEGs with endpoint type `serverless`.\n\n    This flag is not supported for NEGs with endpoint type\n    `private-service-connect`.\n  "
    parser.add_argument('--default-port', type=int, help=help_text)