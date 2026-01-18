from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresEntityTypesListRequest object.

  Fields:
    filter: Lists the EntityTypes that match the filter expression. The
      following filters are supported: * `create_time`: Supports `=`, `!=`,
      `<`, `>`, `>=`, and `<=` comparisons. Values must be in RFC 3339 format.
      * `update_time`: Supports `=`, `!=`, `<`, `>`, `>=`, and `<=`
      comparisons. Values must be in RFC 3339 format. * `labels`: Supports
      key-value equality as well as key presence. Examples: * `create_time >
      \\"2020-01-31T15:30:00.000000Z\\" OR update_time >
      \\"2020-01-31T15:30:00.000000Z\\"` --> EntityTypes created or updated
      after 2020-01-31T15:30:00.000000Z. * `labels.active = yes AND labels.env
      = prod` --> EntityTypes having both (active: yes) and (env: prod)
      labels. * `labels.env: *` --> Any EntityType which has a label with
      'env' as the key.
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending. Supported fields: *
      `entity_type_id` * `create_time` * `update_time`
    pageSize: The maximum number of EntityTypes to return. The service may
      return fewer than this value. If unspecified, at most 1000 EntityTypes
      will be returned. The maximum value is 1000; any value greater than 1000
      will be coerced to 1000.
    pageToken: A page token, received from a previous
      FeaturestoreService.ListEntityTypes call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      FeaturestoreService.ListEntityTypes must match the call that provided
      the page token.
    parent: Required. The resource name of the Featurestore to list
      EntityTypes. Format:
      `projects/{project}/locations/{location}/featurestores/{featurestore}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)