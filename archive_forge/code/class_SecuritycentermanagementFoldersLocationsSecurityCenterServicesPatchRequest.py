from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementFoldersLocationsSecurityCenterServicesPatchRequest(_messages.Message):
    """A
  SecuritycentermanagementFoldersLocationsSecurityCenterServicesPatchRequest
  object.

  Fields:
    name: Identifier. The name of the service Formats: * organizations/{organi
      zation}/locations/{location}/securityCenterServices/{service} *
      folders/{folder}/locations/{location}/securityCenterServices/{service} *
      projects/{project}/locations/{location}/securityCenterServices/{service}
    securityCenterService: A SecurityCenterService resource to be passed as
      the request body.
    updateMask: Required. The list of fields to be updated.
    validateOnly: Optional. When set to true, only validations (including IAM
      checks) will done for the request (service will not be updated). An OK
      response indicates the request is valid while an error response
      indicates the request is invalid. Note that a subsequent request to
      actually update the service could still fail because 1. the state could
      have changed (e.g. IAM permission lost) or 2. A failure occurred while
      trying to delete the module.
  """
    name = _messages.StringField(1, required=True)
    securityCenterService = _messages.MessageField('SecurityCenterService', 2)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)