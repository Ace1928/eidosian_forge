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
def GetCreateInTransitEncryptionConfigMapper(messages, hidden=False):
    """Returns a mapper from text options to the InTransitEncryptionConfig enum.

  Args:
    messages: The message module.
    hidden: Whether the flag should be hidden in the choice_arg.
  """
    help_text = '\nSets the in-transit encryption type for dataplane v2 clusters.\n\n--in-transit-encryption must be one of:\n\n  inter-node-transparent\n    Changes clusters to use transparent, dataplane v2, node-to-node encryption.\n\n  none:\n    Disables dataplane v2 in-transit encryption.\n\n  $ gcloud container clusters create       --in-transit-encryption=inter-node-transparent\n  $ gcloud container clusters create       --in-transit-encryption=none\n'
    return arg_utils.ChoiceEnumMapper('--in-transit-encryption', messages.NetworkConfig.InTransitEncryptionConfigValueValuesEnum, _GetInTransitEncryptionConfigCustomMappings(), hidden=hidden, help_str=help_text)