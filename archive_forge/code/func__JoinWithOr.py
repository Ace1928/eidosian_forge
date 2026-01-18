from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import network_endpoint_groups
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.network_endpoint_groups import flags
from googlecloudsdk.core import log
def _JoinWithOr(strings):
    """Joins strings, for example, into a string like 'A or B' or 'A, B, or C'."""
    if not strings:
        return ''
    elif len(strings) == 1:
        return strings[0]
    elif len(strings) == 2:
        return strings[0] + ' or ' + strings[1]
    else:
        return ', '.join(strings[:-1]) + ', or ' + strings[-1]