from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DataplexFilesetSpec(_messages.Message):
    """Entry specyfication for a Dataplex fileset.

  Fields:
    dataplexSpec: Common Dataplex fields.
  """
    dataplexSpec = _messages.MessageField('GoogleCloudDatacatalogV1DataplexSpec', 1)