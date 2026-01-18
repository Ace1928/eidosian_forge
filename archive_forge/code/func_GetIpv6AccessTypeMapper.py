from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def GetIpv6AccessTypeMapper(messages, hidden=True):
    """Returns a mapper from text options to the Ipv6AccessType enum.

  Args:
    messages: The message module.
    hidden: Whether the flag should be hidden in the choice_arg
  """
    help_text = '\nSets the IPv6 access type for the subnet created by GKE.\n\nIPV6_ACCESS_TYPE must be one of:\n\n  internal\n    Creates a subnet with INTERNAL IPv6 access type.\n\n  external\n    Default value. Creates a subnet with EXTERNAL IPv6 access type.\n\n  $ gcloud container clusters create       --ipv6-access-type=internal\n  $ gcloud container clusters create       --ipv6-access-type=external\n'
    return arg_utils.ChoiceEnumMapper('--ipv6-access-type', messages.IPAllocationPolicy.Ipv6AccessTypeValueValuesEnum, _GetIpv6AccessTypeCustomMappings(), hidden=hidden, help_str=help_text)