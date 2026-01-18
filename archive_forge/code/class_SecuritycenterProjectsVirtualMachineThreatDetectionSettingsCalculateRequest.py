from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsVirtualMachineThreatDetectionSettingsCalculateRequest(_messages.Message):
    """A
  SecuritycenterProjectsVirtualMachineThreatDetectionSettingsCalculateRequest
  object.

  Fields:
    name: Required. The name of the VirtualMachineThreatDetectionSettings to
      calculate. Formats: *
      organizations/{organization}/virtualMachineThreatDetectionSettings *
      folders/{folder}/virtualMachineThreatDetectionSettings *
      projects/{project}/virtualMachineThreatDetectionSettings
  """
    name = _messages.StringField(1, required=True)