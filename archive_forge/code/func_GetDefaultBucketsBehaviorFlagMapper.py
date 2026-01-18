from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def GetDefaultBucketsBehaviorFlagMapper(hidden=False):
    """Gets a mapper for default buckets behavior flag enum value.

  Args:
    hidden: If true, retain help but do not display it.

  Returns:
    A mapper for default buckets behavior flag enum value.
  """
    return arg_utils.ChoiceEnumMapper('--default-buckets-behavior', cloudbuild_util.GetMessagesModule().BuildOptions.DefaultLogsBucketBehaviorValueValuesEnum, include_filter=lambda s: six.text_type(s) != 'UNSPECIFIED', help_str='How default buckets are setup.', hidden=hidden)