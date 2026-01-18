from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1GraphQLOperationConfig(_messages.Message):
    """Binds the resources in a proxy or remote service with the GraphQL
  operation and its associated quota enforcement.

  Fields:
    apiSource: Required. Name of the API proxy endpoint or remote service with
      which the GraphQL operation and quota are associated.
    attributes: Custom attributes associated with the operation.
    operations: Required. List of GraphQL name/operation type pairs for the
      proxy or remote service to which quota will be applied. If only
      operation types are specified, the quota will be applied to all GraphQL
      requests irrespective of the GraphQL name. **Note**: Currently, you can
      specify only a single GraphQLOperation. Specifying more than one will
      cause the operation to fail.
    quota: Quota parameters to be enforced for the resources, methods, and API
      source combination. If none are specified, quota enforcement will not be
      done.
  """
    apiSource = _messages.StringField(1)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 2, repeated=True)
    operations = _messages.MessageField('GoogleCloudApigeeV1GraphQLOperation', 3, repeated=True)
    quota = _messages.MessageField('GoogleCloudApigeeV1Quota', 4)