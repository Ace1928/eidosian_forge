from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesEntitiesPartitionsDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesEntitiesPartitionsDeleteRequest
  object.

  Fields:
    etag: Optional. The etag associated with the partition.
    name: Required. The resource name of the partition. format: projects/{proj
      ect_number}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}/enti
      ties/{entity_id}/partitions/{partition_value_path}. The
      {partition_value_path} segment consists of an ordered sequence of
      partition values separated by "/". All values must be provided.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)