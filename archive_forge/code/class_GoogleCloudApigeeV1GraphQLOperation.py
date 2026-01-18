from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1GraphQLOperation(_messages.Message):
    """Represents the pairing of GraphQL operation types and the GraphQL
  operation name.

  Fields:
    operation: GraphQL operation name. The name and operation type will be
      used to apply quotas. If no name is specified, the quota will be applied
      to all GraphQL operations irrespective of their operation names in the
      payload.
    operationTypes: Required. GraphQL operation types. Valid values include
      `query` or `mutation`. **Note**: Apigee does not currently support
      `subscription` types.
  """
    operation = _messages.StringField(1)
    operationTypes = _messages.StringField(2, repeated=True)