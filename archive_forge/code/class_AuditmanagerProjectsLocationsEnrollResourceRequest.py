from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditmanagerProjectsLocationsEnrollResourceRequest(_messages.Message):
    """A AuditmanagerProjectsLocationsEnrollResourceRequest object.

  Fields:
    enrollResourceRequest: A EnrollResourceRequest resource to be passed as
      the request body.
    scope: Required. The resource to be enrolled to the audit manager. Scope
      format should be resource_type/resource_identifier Eg:
      projects/{project-id}/locations/{location}, folders/{folder-
      id}/locations/{location}
  """
    enrollResourceRequest = _messages.MessageField('EnrollResourceRequest', 1)
    scope = _messages.StringField(2, required=True)