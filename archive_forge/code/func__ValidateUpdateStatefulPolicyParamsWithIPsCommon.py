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
def _ValidateUpdateStatefulPolicyParamsWithIPsCommon(current_interface_names, update_flag_name, remove_flag_name, update_ips, remove_ips, ip_type_name):
    """Check stateful properties of update request."""
    update_interface_names = []
    if update_ips:
        ValidateStatefulIPDicts(update_ips, update_flag_name)
        for stateful_ip in update_ips:
            update_interface_names.append(stateful_ip.get('interface-name', STATEFUL_IP_DEFAULT_INTERFACE_NAME))
    remove_interface_names = remove_ips or []
    if any((remove_interface_names.count(x) > 1 for x in remove_interface_names)):
        raise exceptions.InvalidArgumentException(parameter_name='update', message='When removing stateful {} IPs from Stateful Policy, please provide each network interface name exactly once.'.format(ip_type_name))
    update_set = set(update_interface_names)
    remove_set = set(remove_interface_names)
    intersection = update_set.intersection(remove_set)
    if intersection:
        raise exceptions.InvalidArgumentException(parameter_name='update', message='You cannot simultaneously add and remove the same interface {} to stateful {} IPs in Stateful Policy.'.format(six.text_type(intersection), ip_type_name))
    not_current_interface_names = remove_set - current_interface_names
    if not_current_interface_names:
        raise exceptions.InvalidArgumentException(parameter_name='update', message='Interfaces [{}] are not currently set as stateful {} IPs, so they cannot be removed from Stateful Policy.'.format(six.text_type(not_current_interface_names), ip_type_name))