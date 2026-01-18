from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1ImportTaxonomiesRequest(_messages.Message):
    """Request message for ImportTaxonomies.

  Fields:
    inlineSource: Inline source used for taxonomies to be imported.
  """
    inlineSource = _messages.MessageField('GoogleCloudDatacatalogV1beta1InlineSource', 1)