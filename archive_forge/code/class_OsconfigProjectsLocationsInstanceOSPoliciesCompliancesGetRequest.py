from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsconfigProjectsLocationsInstanceOSPoliciesCompliancesGetRequest(_messages.Message):
    """A OsconfigProjectsLocationsInstanceOSPoliciesCompliancesGetRequest
  object.

  Fields:
    name: Required. API resource name for instance OS policies compliance
      resource. Format: `projects/{project}/locations/{location}/instanceOSPol
      iciesCompliances/{instance}` For `{project}`, either Compute Engine
      project-number or project-id can be provided. For `{instance}`, either
      Compute Engine VM instance-id or instance-name can be provided.
  """
    name = _messages.StringField(1, required=True)