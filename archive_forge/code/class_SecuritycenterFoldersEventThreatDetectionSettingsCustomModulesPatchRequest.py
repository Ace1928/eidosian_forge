from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesPatchRequest(_messages.Message):
    """A
  SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesPatchRequest
  object.

  Fields:
    eventThreatDetectionCustomModule: A EventThreatDetectionCustomModule
      resource to be passed as the request body.
    name: Immutable. The resource name of the Event Threat Detection custom
      module. Its format is: * "organizations/{organization}/eventThreatDetect
      ionSettings/customModules/{module}". *
      "folders/{folder}/eventThreatDetectionSettings/customModules/{module}".
      * "projects/{project}/eventThreatDetectionSettings/customModules/{module
      }".
    updateMask: The list of fields to be updated. If empty all mutable fields
      will be updated.
  """
    eventThreatDetectionCustomModule = _messages.MessageField('EventThreatDetectionCustomModule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)