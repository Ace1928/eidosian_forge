from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
class SubnetOption(enum.Enum):
    ALL_RANGES = 0
    PRIMARY_RANGES = 1
    CUSTOM_RANGES = 2