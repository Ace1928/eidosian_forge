from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1FilesetSpec(_messages.Message):
    """Specification that applies to a fileset. Valid only for entries with the
  'FILESET' type.

  Fields:
    dataplexFileset: Fields specific to a Dataplex fileset and present only in
      the Dataplex fileset entries.
  """
    dataplexFileset = _messages.MessageField('GoogleCloudDatacatalogV1DataplexFilesetSpec', 1)