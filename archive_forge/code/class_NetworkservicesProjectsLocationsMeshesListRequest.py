from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMeshesListRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMeshesListRequest object.

  Fields:
    pageSize: Maximum number of Meshes to return per call.
    pageToken: The value returned by the last `ListMeshesResponse` Indicates
      that this is a continuation of a prior `ListMeshes` call, and that the
      system should return the next page of data.
    parent: Required. The project and location from which the Meshes should be
      listed, specified in the format `projects/*/locations/global`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)