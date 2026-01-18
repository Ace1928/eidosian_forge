from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsAlertPoliciesCreateRequest(_messages.Message):
    """A MonitoringProjectsAlertPoliciesCreateRequest object.

  Fields:
    alertPolicy: A AlertPolicy resource to be passed as the request body.
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) in which to
      create the alerting policy. The format is:
      projects/[PROJECT_ID_OR_NUMBER] Note that this field names the parent
      container in which the alerting policy will be written, not the name of
      the created policy. |name| must be a host project of a Metrics Scope,
      otherwise INVALID_ARGUMENT error will return. The alerting policy that
      is returned will have a name that contains a normalized representation
      of this name as a prefix but adds a suffix of the form
      /alertPolicies/[ALERT_POLICY_ID], identifying the policy in the
      container.
  """
    alertPolicy = _messages.MessageField('AlertPolicy', 1)
    name = _messages.StringField(2, required=True)