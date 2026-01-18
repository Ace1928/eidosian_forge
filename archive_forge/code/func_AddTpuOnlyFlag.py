from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTpuOnlyFlag(parser, help_text_override=None):
    help_text = '      Do not allocate a VM, only allocate a TPU. To be used after the command has been run with a --vm-only flag\n      and the user is ready to run on a TPU. Ensure that the name matches the name passed in when creating with the --vm-only flag.\n      '
    return parser.add_argument('--tpu-only', action='store_true', required=False, default=False, help=help_text_override or help_text)