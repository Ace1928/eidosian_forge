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
def _GetValidScopesErrorMessage(network_endpoint_type, valid_scopes):
    valid_scopes_error_message = ''
    if network_endpoint_type in valid_scopes:
        valid_scopes_error_message = ' Type {0} must be specified in the {1} scope.'.format(network_endpoint_type, _JoinWithOr(valid_scopes[network_endpoint_type]))
    return valid_scopes_error_message