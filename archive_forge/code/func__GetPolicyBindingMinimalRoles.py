from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def _GetPolicyBindingMinimalRoles(gcloud_insight):
    """Returns minimal roles extracted from the IAM policy binding insight.

  Args:
    gcloud_insight: Insight returned by the recommender API.

  Returns: A list of strings. Empty if no minimal roles were found.
  """
    minimal_roles = []
    for additional_property in gcloud_insight.content.additionalProperties:
        if additional_property.key == 'risk':
            for p in additional_property.value.object_value.properties:
                if p.key == 'usageAtRisk':
                    for f in p.value.object_value.properties:
                        if f.key == 'iamPolicyUtilization':
                            for iam_p in f.value.object_value.properties:
                                if iam_p.key == 'minimalRoles':
                                    for role in iam_p.value.array_value.entries:
                                        minimal_roles.append(role.string_value)
    return minimal_roles