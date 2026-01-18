from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetRequestedFeatures(messages, requested_features_arg):
    """Converts the requested-features flag to a list of message enums.

  Args:
    messages: The API messages holder.
    requested_features_arg: A list of the interconnect feature type flag values.

  Returns:
    A list of RequestedFeaturesValueListEntryValuesEnum values, or None if
    absent.
  """
    if not requested_features_arg:
        return []
    features = list(filter(None, [GetRequestedFeature(messages, f) for f in requested_features_arg]))
    return list(collections.OrderedDict.fromkeys(features))