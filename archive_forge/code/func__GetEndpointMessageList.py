from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.util.apis import arg_utils
def _GetEndpointMessageList(self, endpoints):
    """Convert endpoints to a list which can be passed in a request."""
    output_list = []
    for arg_endpoint in endpoints:
        message_endpoint = self.messages.NetworkEndpoint()
        if 'instance' in arg_endpoint:
            message_endpoint.instance = arg_endpoint.get('instance')
        if 'ip' in arg_endpoint:
            message_endpoint.ipAddress = arg_endpoint.get('ip')
        if 'ipv6' in arg_endpoint:
            message_endpoint.ipv6Address = arg_endpoint.get('ipv6')
        if 'port' in arg_endpoint:
            message_endpoint.port = arg_endpoint.get('port')
        if 'fqdn' in arg_endpoint:
            message_endpoint.fqdn = arg_endpoint.get('fqdn')
        if 'client-port' in arg_endpoint:
            message_endpoint.clientPort = arg_endpoint.get('client-port')
        output_list.append(message_endpoint)
    return output_list