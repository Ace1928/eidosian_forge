from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementOrganizationsLocationsEventThreatDetectionCustomModulesDeleteRequest(_messages.Message):
    """A SecuritycentermanagementOrganizationsLocationsEventThreatDetectionCust
  omModulesDeleteRequest object.

  Fields:
    name: Required. The resource name of the ETD custom module. Its format is:
      * "organizations/{organization}/locations/{location}/eventThreatDetectio
      nCustomModules/{event_threat_detection_custom_module}". * "folders/{fold
      er}/locations/{location}/eventThreatDetectionCustomModules/{event_threat
      _detection_custom_module}". * "projects/{project}/locations/{location}/e
      ventThreatDetectionCustomModules/{event_threat_detection_custom_module}"
      .
    validateOnly: Optional. When set to true, only validations (including IAM
      checks) will done for the request (module will not be deleted). An OK
      response indicates the request is valid while an error response
      indicates the request is invalid. Note that a subsequent request to
      actually delete the module could still fail because 1. the state could
      have changed (e.g. IAM permission lost) or 2. A failure occurred while
      trying to delete the module.
  """
    name = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)