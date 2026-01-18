from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.compute.sole_tenancy.node_templates import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def GetAccelerators(args, messages):
    """Returns list of messages with accelerators for the instance."""
    if args.accelerator:
        accelerator_type = args.accelerator['type']
        accelerator_count = int(args.accelerator.get('count', 4))
        return CreateAcceleratorConfigMessages(messages, accelerator_type, accelerator_count)
    return []