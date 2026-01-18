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
def _CheckNodeConfigValueType(name, value, value_type):
    """Check whether the config option has the expected value type.

  Args:
    name: The name of the config option to be checked.
    value: The value of the config option to be checked.
    value_type: The expected value type (e.g., str, bool, dict).

  Raises:
    NodeConfigError: if value is not of value_type.
  """
    if not isinstance(value, value_type):
        raise NodeConfigError('value of "{0}" must be {1}'.format(name, value_type.__name__))