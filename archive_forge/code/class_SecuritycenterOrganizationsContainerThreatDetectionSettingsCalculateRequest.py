from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsContainerThreatDetectionSettingsCalculateRequest(_messages.Message):
    """A
  SecuritycenterOrganizationsContainerThreatDetectionSettingsCalculateRequest
  object.

  Fields:
    name: Required. The name of the ContainerThreatDetectionSettings to
      calculate. Formats: *
      organizations/{organization}/containerThreatDetectionSettings *
      folders/{folder}/containerThreatDetectionSettings *
      projects/{project}/containerThreatDetectionSettings * projects/{project}
      /locations/{location}/clusters/{cluster}/containerThreatDetectionSetting
      s
  """
    name = _messages.StringField(1, required=True)