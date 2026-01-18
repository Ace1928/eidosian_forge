from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1UpdateSecurityIncidentRequest(_messages.Message):
    """Request for UpdateSecurityIncident.

  Fields:
    securityIncident: Required. The security incident to update. Must contain
      all existing populated fields of the current incident.
    updateMask: Required. The list of fields to update. Allowed fields are:
      LINT.IfChange(allowed_update_fields_comment) - observability
      LINT.ThenChange()
  """
    securityIncident = _messages.MessageField('GoogleCloudApigeeV1SecurityIncident', 1)
    updateMask = _messages.StringField(2)