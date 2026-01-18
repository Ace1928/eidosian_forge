from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ImportTaxonomiesRequest(_messages.Message):
    """Request message for ImportTaxonomies.

  Fields:
    crossRegionalSource: Cross-regional source taxonomy to import.
    inlineSource: Inline source taxonomy to import.
  """
    crossRegionalSource = _messages.MessageField('GoogleCloudDatacatalogV1CrossRegionalSource', 1)
    inlineSource = _messages.MessageField('GoogleCloudDatacatalogV1InlineSource', 2)