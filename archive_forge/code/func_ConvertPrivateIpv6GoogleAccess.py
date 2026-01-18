from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.networks.subnets import flags
import six
def ConvertPrivateIpv6GoogleAccess(choice):
    """Return PrivateIpv6GoogleAccess enum defined in mixer.

  Args:
    choice: Enum value of PrivateIpv6GoogleAccess defined in gcloud.
  """
    choices_to_enum = {'DISABLE': 'DISABLE_GOOGLE_ACCESS', 'ENABLE_BIDIRECTIONAL_ACCESS': 'ENABLE_BIDIRECTIONAL_ACCESS_TO_GOOGLE', 'ENABLE_OUTBOUND_VM_ACCESS': 'ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE'}
    return choices_to_enum.get(choice)