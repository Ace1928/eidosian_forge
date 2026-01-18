from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsServicesRevisionsListRequest(_messages.Message):
    """A RunProjectsLocationsServicesRevisionsListRequest object.

  Fields:
    pageSize: Maximum number of revisions to return in this call.
    pageToken: A page token received from a previous call to ListRevisions.
      All other parameters must match.
    parent: Required. The Service from which the Revisions should be listed.
      To list all Revisions across Services, use "-" instead of Service name.
      Format: projects/{project}/locations/{location}/services/{service}
    showDeleted: If true, returns deleted (but unexpired) resources along with
      active ones.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)