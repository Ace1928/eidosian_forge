from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def _GetResourceRiskReasons(gcloud_insight):
    """Extracts a list of string reasons from the resource change insight.

  Args:
    gcloud_insight: Insight object returned by the recommender API.

  Returns:
    A list of strings. If no reasons could be found, then returns empty list.
    The number of reasons is limited by _MAX_NUMBER_OF_REASONS, and the last
    reason indicates how many more reasons there are if applicable.
  """
    reasons = []
    num_reasons = 0
    last_reason = ''
    for additional_property in gcloud_insight.content.additionalProperties:
        if additional_property.key == 'importance':
            for p in additional_property.value.object_value.properties:
                if p.key == 'detailedReasons':
                    for reason in p.value.array_value.entries:
                        num_reasons += 1
                        if num_reasons < _MAX_NUMBER_OF_REASONS:
                            reasons.append(reason.string_value)
                        elif num_reasons == _MAX_NUMBER_OF_REASONS:
                            last_reason = reason.string_value
    if num_reasons > _MAX_NUMBER_OF_REASONS:
        last_reason = '{} other importance indicators.'.format(num_reasons - _MAX_NUMBER_OF_REASONS + 1)
    if num_reasons >= _MAX_NUMBER_OF_REASONS:
        reasons.append(last_reason)
    return reasons