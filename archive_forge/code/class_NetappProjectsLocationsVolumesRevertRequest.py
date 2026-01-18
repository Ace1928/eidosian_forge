from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesRevertRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesRevertRequest object.

  Fields:
    name: Required. The resource name of the volume, in the format of
      projects/{project_id}/locations/{location}/volumes/{volume_id}.
    revertVolumeRequest: A RevertVolumeRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    revertVolumeRequest = _messages.MessageField('RevertVolumeRequest', 2)