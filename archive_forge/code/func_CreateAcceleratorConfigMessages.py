from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.compute.sole_tenancy.node_templates import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def CreateAcceleratorConfigMessages(msgs, accelerator_type, accelerator_count):
    """Returns a list of accelerator config messages.

  Args:
    msgs: tracked GCE API messages.
    accelerator_type: reference to the accelerator type.
    accelerator_count: number of accelerators to attach to the VM.

  Returns:
    a list of accelerator config message that specifies the type and number of
    accelerators to attach to an instance.
  """
    accelerator_config = msgs.AcceleratorConfig(acceleratorType=accelerator_type, acceleratorCount=accelerator_count)
    return [accelerator_config]