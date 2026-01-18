from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log as sdk_log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
def ExtractLogId(log_resource):
    """Extracts only the log id and restore original slashes.

  Args:
    log_resource: The full log uri e.g projects/my-projects/logs/my-log.

  Returns:
    A log id that can be used in other commands.
  """
    log_id = log_resource.split('/logs/', 1)[1]
    return log_id.replace('%2F', '/')