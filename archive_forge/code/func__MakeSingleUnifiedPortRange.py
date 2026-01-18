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
def _MakeSingleUnifiedPortRange(arg_port_range, range_list_from_arg_ports):
    """Reconciles the deprecated --port-range arg with ranges from --ports arg."""
    if arg_port_range:
        log.warning('The --port-range flag is deprecated. Use equivalent --ports=%s flag.', arg_port_range)
        return six.text_type(arg_port_range)
    elif range_list_from_arg_ports:
        range_list = _UnifyPortRangeFromListOfRanges(range_list_from_arg_ports)
        return six.text_type(range_list) if range_list else None