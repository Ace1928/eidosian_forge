from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CatalogIndexedFieldConfig(_messages.Message):
    """CatalogIndexedFieldConfig configures the asset type's metadata fields
  that need to be included in catalog's search result.

  Fields:
    expression: The expression that points to the indexed field in the asset.
      If it's empty, the default is metadata.key, and the key will not be used
      as search operator.
  """
    expression = _messages.StringField(1)