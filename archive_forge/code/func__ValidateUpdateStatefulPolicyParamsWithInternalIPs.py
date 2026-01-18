from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def _ValidateUpdateStatefulPolicyParamsWithInternalIPs(args, current_stateful_policy):
    """Check stateful internal IPs properties of update request."""
    current_interface_names = set(managed_instance_groups_utils.GetInterfaceNamesFromStatefulPolicyForInternalIPs(current_stateful_policy))
    _ValidateUpdateStatefulPolicyParamsWithIPsCommon(current_interface_names, '--stateful-internal-ip', '--remove-stateful-internal-ips', args.stateful_internal_ip, args.remove_stateful_internal_ips, 'internal')