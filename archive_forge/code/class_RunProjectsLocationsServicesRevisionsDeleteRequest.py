from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsServicesRevisionsDeleteRequest(_messages.Message):
    """A RunProjectsLocationsServicesRevisionsDeleteRequest object.

  Fields:
    etag: A system-generated fingerprint for this version of the resource.
      This may be used to detect modification conflict during updates.
    name: Required. The name of the Revision to delete. Format: projects/{proj
      ect}/locations/{location}/services/{service}/revisions/{revision}
    validateOnly: Indicates that the request should be validated without
      actually deleting any resources.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)