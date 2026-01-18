from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import forwarding_rules_utils as utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions as fw_exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.core import log
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _UnifyPortRangeFromListOfRanges(ports_range_list):
    """Return a single port range by combining a list of port ranges."""
    if not ports_range_list:
        return (None, None)
    ports = sorted(ports_range_list)
    combined_port_range = ports.pop(0)
    for port_range in ports_range_list:
        try:
            combined_port_range = combined_port_range.Combine(port_range)
        except arg_parsers.Error:
            raise exceptions.InvalidArgumentException('--ports', 'Must specify consecutive ports at this time.')
    return combined_port_range