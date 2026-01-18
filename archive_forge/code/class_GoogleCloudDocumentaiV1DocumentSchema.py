from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentSchema(_messages.Message):
    """The schema defines the output of the processed document by a processor.

  Fields:
    description: Description of the schema.
    displayName: Display name to show to users.
    entityTypes: Entity types of the schema.
    metadata: Metadata of the schema.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    entityTypes = _messages.MessageField('GoogleCloudDocumentaiV1DocumentSchemaEntityType', 3, repeated=True)
    metadata = _messages.MessageField('GoogleCloudDocumentaiV1DocumentSchemaMetadata', 4)