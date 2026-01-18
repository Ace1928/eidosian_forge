from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesEntitiesPartitionsListRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesEntitiesPartitionsListRequest
  object.

  Fields:
    filter: Optional. Filter the partitions returned to the caller using a key
      value pair expression. Supported operators and syntax: logic operators:
      AND, OR comparison operators: <, >, >=, <= ,=, != LIKE operators: The
      right hand of a LIKE operator supports "." and "*" for wildcard
      searches, for example "value1 LIKE ".*oo.*" parenthetical grouping: (
      )Sample filter expression: `?filter="key1 < value1 OR key2 >
      value2"Notes: Keys to the left of operators are case insensitive.
      Partition results are sorted first by creation time, then by
      lexicographic order. Up to 20 key value filter pairs are allowed, but
      due to performance considerations, only the first 10 will be used as a
      filter.
    pageSize: Optional. Maximum number of partitions to return. The service
      may return fewer than this value. If unspecified, 100 partitions will be
      returned by default. The maximum page size is 500; larger values will
      will be truncated to 500.
    pageToken: Optional. Page token received from a previous ListPartitions
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to ListPartitions must match the call that
      provided the page token.
    parent: Required. The resource name of the parent entity: projects/{projec
      t_number}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}/entiti
      es/{entity_id}.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)