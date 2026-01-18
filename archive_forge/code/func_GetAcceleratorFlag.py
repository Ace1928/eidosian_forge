from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetAcceleratorFlag(custom_name=None):
    """Gets the --accelerator flag."""
    help_text = '  Manage the configuration of the type and number of accelerator cards attached.\n  *count*::: The number of accelerators to attach to each instance in the reservation.\n  *type*::: The specific type (e.g. `nvidia-tesla-k80` for nVidia Tesla K80) of\n  accelerator to attach to instances in the reservation. Use `gcloud compute accelerator-types list`\n  to learn about all available accelerator types.\n  '
    return base.Argument(custom_name if custom_name else '--accelerator', type=arg_parsers.ArgDict(spec={'count': int, 'type': str}, required_keys=['count', 'type']), action='append', help=help_text)