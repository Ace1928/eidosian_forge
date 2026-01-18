from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyConsentsRequest(_messages.Message):
    """Request to apply the Consent resources for the specified FHIR store.

  Fields:
    patientScope: Optional. Scope down to a list of patients.
    timeRange: Optional. Scope down to patients whose most recent consent
      changes are in the time range. Can only be used with a versioning store
      (i.e. when disable_resource_versioning is set to false).
    validateOnly: Optional. If true, the method only validates Consent
      resources to make sure they are supported. When the operation completes,
      ApplyConsentsResponse is returned where `consent_apply_success` and
      `consent_apply_failure` indicate supported and unsupported (or invalid)
      Consent resources, respectively. Otherwise, the method propagates the
      aggregate consensual information to the patient's resources. Upon
      success, `affected_resources` in the ApplyConsentsResponse indicates the
      number of resources that may have consensual access changed.
  """
    patientScope = _messages.MessageField('PatientScope', 1)
    timeRange = _messages.MessageField('TimeRange', 2)
    validateOnly = _messages.BooleanField(3)