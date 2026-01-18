from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsDeleteRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsDele
  teRequest object.

  Fields:
    etag: Optional. If set, the request is rejected if the latest version of
      the workstation configuration on the server does not have this ETag.
    force: Optional. If set, any workstations in the workstation configuration
      are also deleted. Otherwise, the request works only if the workstation
      configuration has no workstations.
    name: Required. Name of the workstation configuration to delete.
    validateOnly: Optional. If set, validate the request and preview the
      review, but do not actually apply it.
  """
    etag = _messages.StringField(1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)