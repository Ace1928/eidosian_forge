from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddRemoveEndpoint(endpoint_group, endpoint_spec, support_ipv6, support_port_mapping_neg):
    """Adds remove endpoint argument for updating network endpoint groups."""
    help_text = '          The network endpoint to detach from the network endpoint group. Keys\n          used depend on the endpoint type of the NEG.\n\n          `gce-vm-ip-port`\n\n              *instance* - Required name of instance whose endpoint(s) to\n              detach. If the IP address is unset, all endpoints for the\n              instance in the NEG are detached.\n\n              *ip* - Optional IPv4 address of the network endpoint to detach.\n              If specified port must be provided as well.\n  '
    if support_ipv6:
        help_text += '\n              *ipv6* - Optional IPv6 address of the network endpoint to detach.\n              If specified port must be provided as well.\n    '
    help_text += '\n              *port* - Optional port of the network endpoint to detach.\n    '
    if support_port_mapping_neg:
        help_text += '\n              *client-port* - Optional client port, only for port mapping NEGs.\n               '
    help_text += '\n          `internet-ip-port`\n\n              *ip* - Required IPv4 address of the network endpoint to detach.\n  '
    if support_ipv6:
        help_text += '\n              *ipv6* - Required IPv6 address of the network endpoint to detach.\n\n              At least one of the ip and ipv6 must be specified.\n    '
    help_text += '\n              *port* - Optional port of the network endpoint to detach if the\n              endpoint has a port specified.\n\n          `internet-fqdn-port`\n\n              *fqdn* - Required fully qualified domain name of the endpoint to\n              detach.\n\n              *port* - Optional port of the network endpoint to detach if the\n              endpoint has a port specified.\n\n          `non-gcp-private-ip-port`\n\n              *ip* - Required IPv4 address of the network endpoint to detach.\n    '
    if support_ipv6:
        help_text += '\n              *ipv6* - Required IPv6 address of the network endpoint to detach.\n\n              At least one of the ip and ipv6 must be specified.\n      '
    help_text += "\n              *port* - Required port of the network endpoint to detach unless\n              NEG default port is set.\n\n          `gce-vm-ip`\n\n              *instance* - Required name of instance with endpoints to\n              detach. If the IP address is unset, all endpoints for the\n              instance in the NEG are detached.\n\n              *ip* - Optional IP address of the network endpoint to attach. The\n              IP address must be the VM's network interface's primary IP\n              address. If not specified, the primary NIC address is used.\n  "
    endpoint_group.add_argument('--remove-endpoint', action='append', type=arg_parsers.ArgDict(spec=endpoint_spec), help=help_text)