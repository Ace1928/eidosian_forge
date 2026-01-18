from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsDeleteRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWork
  stationsDeleteRequest object.

  Fields:
    etag: Optional. If set, the request will be rejected if the latest version
      of the workstation on the server does not have this ETag.
    name: Required. Name of the workstation to delete.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually apply it.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)