from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMeshesGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMeshesGetRequest object.

  Fields:
    name: Required. A name of the Mesh to get. Must be in the format
      `projects/*/locations/global/meshes/*`.
  """
    name = _messages.StringField(1, required=True)