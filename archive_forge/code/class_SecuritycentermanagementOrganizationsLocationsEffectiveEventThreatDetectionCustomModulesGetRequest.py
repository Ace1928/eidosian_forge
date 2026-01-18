from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementOrganizationsLocationsEffectiveEventThreatDetectionCustomModulesGetRequest(_messages.Message):
    """A SecuritycentermanagementOrganizationsLocationsEffectiveEventThreatDete
  ctionCustomModulesGetRequest object.

  Fields:
    name: Required. The resource name of the ETD custom module. Its format is:
      * "organizations/{organization}/locations/{location}/effectiveEventThrea
      tDetectionCustomModules/{effective_event_threat_detection_custom_module}
      ". * "folders/{folder}/locations/{location}/effectiveEventThreatDetectio
      nCustomModules/{effective_event_threat_detection_custom_module}". * "pro
      jects/{project}/locations/{location}/effectiveEventThreatDetectionCustom
      Modules/{effective_event_threat_detection_custom_module}".
  """
    name = _messages.StringField(1, required=True)