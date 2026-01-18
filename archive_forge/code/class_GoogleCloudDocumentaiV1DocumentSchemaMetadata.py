from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentSchemaMetadata(_messages.Message):
    """Metadata for global schema behavior.

  Fields:
    documentAllowMultipleLabels: If true, on a given page, there can be
      multiple `document` annotations covering it.
    documentSplitter: If true, a `document` entity type can be applied to
      subdocument (splitting). Otherwise, it can only be applied to the entire
      document (classification).
    prefixedNamingOnProperties: If set, all the nested entities must be
      prefixed with the parents.
    skipNamingValidation: If set, we will skip the naming format validation in
      the schema. So the string values in `DocumentSchema.EntityType.name` and
      `DocumentSchema.EntityType.Property.name` will not be checked.
  """
    documentAllowMultipleLabels = _messages.BooleanField(1)
    documentSplitter = _messages.BooleanField(2)
    prefixedNamingOnProperties = _messages.BooleanField(3)
    skipNamingValidation = _messages.BooleanField(4)