from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExportFeatureValuesRequestSnapshotExport(_messages.Message):
    """Describes exporting the latest Feature values of all entities of the
  EntityType between [start_time, snapshot_time].

  Fields:
    snapshotTime: Exports Feature values as of this timestamp. If not set,
      retrieve values as of now. Timestamp, if present, must not have higher
      than millisecond precision.
    startTime: Excludes Feature values with feature generation timestamp
      before this timestamp. If not set, retrieve oldest values kept in
      Feature Store. Timestamp, if present, must not have higher than
      millisecond precision.
  """
    snapshotTime = _messages.StringField(1)
    startTime = _messages.StringField(2)