from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesDeleteRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesDeleteRequest object.

  Fields:
    force: If this field is set as true, CCFE will not block the volume
      resource deletion even if it has any snapshots resource. (Otherwise, the
      request will only work if the volume has no snapshots.)
    name: Required. Name of the volume
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)