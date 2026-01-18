from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1InlineSource(_messages.Message):
    """Inline source used for taxonomies import.

  Fields:
    taxonomies: Required. Taxonomies to be imported.
  """
    taxonomies = _messages.MessageField('GoogleCloudDatacatalogV1beta1SerializedTaxonomy', 1, repeated=True)