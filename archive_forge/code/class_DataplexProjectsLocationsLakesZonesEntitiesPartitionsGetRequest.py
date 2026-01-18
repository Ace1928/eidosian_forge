from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesEntitiesPartitionsGetRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesEntitiesPartitionsGetRequest
  object.

  Fields:
    name: Required. The resource name of the partition: projects/{project_numb
      er}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}/entities/{en
      tity_id}/partitions/{partition_value_path}. The {partition_value_path}
      segment consists of an ordered sequence of partition values separated by
      "/". All values must be provided.
  """
    name = _messages.StringField(1, required=True)