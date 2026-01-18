from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsAutoscalingPoliciesDeleteRequest(_messages.Message):
    """A DataprocProjectsLocationsAutoscalingPoliciesDeleteRequest object.

  Fields:
    name: Required. The "resource name" of the autoscaling policy, as
      described in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.autoscalingPolicies.delete, the resource name of the
      policy has the following format:
      projects/{project_id}/regions/{region}/autoscalingPolicies/{policy_id}
      For projects.locations.autoscalingPolicies.delete, the resource name of
      the policy has the following format: projects/{project_id}/locations/{lo
      cation}/autoscalingPolicies/{policy_id}
  """
    name = _messages.StringField(1, required=True)