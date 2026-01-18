from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import scaled_integer
import six
def GetServerBindingMapperFlag(messages):
    """Helper to get a choice flag from server binding type enum."""
    return arg_utils.ChoiceEnumMapper('--server-binding', messages.ServerBinding.TypeValueValuesEnum, custom_mappings={'RESTART_NODE_ON_ANY_SERVER': ('restart-node-on-any-server', 'Nodes using this template will restart on any physical server following a maintenance event.'), 'RESTART_NODE_ON_MINIMAL_SERVERS': ('restart-node-on-minimal-servers', 'Nodes using this template will restart on the same physical server following a\nmaintenance event, instead of being live migrated to or restarted on a new\nphysical server. This means that VMs on such nodes will experience outages while\nmaintenance is applied. This option may be useful if you are using software\nlicenses tied to the underlying server characteristics such as physical sockets\nor cores, to avoid the need for additional licenses when maintenance occurs.\n\nNote that in some cases, Google Compute Engine may need to move your VMs to a\nnew underlying server. During these situations your VMs will be restarted on a\nnew physical server and assigned a new sole tenant physical server ID.')}, help_str='The server binding policy for nodes using this template, which determines where the nodes should restart following a maintenance event.', default='restart-node-on-any-server')