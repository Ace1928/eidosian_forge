from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSchemasCreateRequest(_messages.Message):
    """A PubsubProjectsSchemasCreateRequest object.

  Fields:
    parent: Required. The name of the project in which to create the schema.
      Format is `projects/{project-id}`.
    schema: A Schema resource to be passed as the request body.
    schemaId: The ID to use for the schema, which will become the final
      component of the schema's resource name. See
      https://cloud.google.com/pubsub/docs/pubsub-basics#resource_names for
      resource name constraints.
  """
    parent = _messages.StringField(1, required=True)
    schema = _messages.MessageField('Schema', 2)
    schemaId = _messages.StringField(3)