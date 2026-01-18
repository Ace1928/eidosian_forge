from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import scaled_integer
import six
def _BinarySize(default_unit, lower_bound=None, upper_bound=None):
    """Parses the value as a binary size converted to the default unit."""
    bytes_per_unit = scaled_integer.GetBinaryUnitSize(default_unit)

    def _Parse(value):
        value = value.lower()
        size = arg_parsers.BinarySize(lower_bound=lower_bound, upper_bound=upper_bound, default_unit=default_unit)(value)
        converted_size = size // bytes_per_unit
        return converted_size
    return _Parse