from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditmanagerFoldersLocationsResourceEnrollmentStatusesGetRequest(_messages.Message):
    """A AuditmanagerFoldersLocationsResourceEnrollmentStatusesGetRequest
  object.

  Fields:
    name: Required. Format folders/{folder}/locations/{location}/resourceEnrol
      lmentStatuses/{resource_enrollment_status}, projects/{project}/locations
      /{location}/resourceEnrollmentStatuses/{resource_enrollment_status}
  """
    name = _messages.StringField(1, required=True)