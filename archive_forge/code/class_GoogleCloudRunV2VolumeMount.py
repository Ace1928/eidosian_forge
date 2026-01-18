from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2VolumeMount(_messages.Message):
    """VolumeMount describes a mounting of a Volume within a container.

  Fields:
    mountPath: Required. Path within the container at which the volume should
      be mounted. Must not contain ':'. For Cloud SQL volumes, it can be left
      empty, or must otherwise be `/cloudsql`. All instances defined in the
      Volume will be available as `/cloudsql/[instance]`. For more information
      on Cloud SQL volumes, visit
      https://cloud.google.com/sql/docs/mysql/connect-run
    name: Required. This must match the Name of a Volume.
  """
    mountPath = _messages.StringField(1)
    name = _messages.StringField(2)