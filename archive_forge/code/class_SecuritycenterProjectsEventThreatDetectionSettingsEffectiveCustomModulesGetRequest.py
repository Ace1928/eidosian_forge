from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsEventThreatDetectionSettingsEffectiveCustomModulesGetRequest(_messages.Message):
    """A SecuritycenterProjectsEventThreatDetectionSettingsEffectiveCustomModul
  esGetRequest object.

  Fields:
    name: Required. The resource name of the effective Event Threat Detection
      custom module. Its format is: * "organizations/{organization}/eventThrea
      tDetectionSettings/effectiveCustomModules/{module}". * "folders/{folder}
      /eventThreatDetectionSettings/effectiveCustomModules/{module}". * "proje
      cts/{project}/eventThreatDetectionSettings/effectiveCustomModules/{modul
      e}".
  """
    name = _messages.StringField(1, required=True)