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
def GetCreateStackTypeMapper(messages, hidden=False):
    """Returns a mapper from text options to the StackType enum.

  Args:
    messages: The message module.
    hidden: Whether the flag should be hidden in the choice_arg
  """
    help_text = '\nSets the stack type for the cluster nodes and pods.\n\nSTACK_TYPE must be one of:\n\n  ipv4\n    Default value. Creates IPv4 single stack clusters.\n\n  ipv4-ipv6\n    Creates dual stack clusters.\n\n  $ gcloud container clusters create       --stack-type=ipv4\n  $ gcloud container clusters create       --stack-type=ipv4-ipv6\n'
    return arg_utils.ChoiceEnumMapper('--stack-type', messages.IPAllocationPolicy.StackTypeValueValuesEnum, _GetStackTypeCustomMappings(), hidden=hidden, help_str=help_text)