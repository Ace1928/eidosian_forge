from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExistenceFilter(_messages.Message):
    """A digest of all the documents that match a given target.

  Fields:
    count: The total count of documents that match target_id. If different
      from the count of documents in the client that match, the client must
      manually determine which documents no longer match the target. The
      client can use the `unchanged_names` bloom filter to assist with this
      determination by testing ALL the document names against the filter; if
      the document name is NOT in the filter, it means the document no longer
      matches the target.
    targetId: The target ID to which this filter applies.
    unchangedNames: A bloom filter that, despite its name, contains the UTF-8
      byte encodings of the resource names of ALL the documents that match
      target_id, in the form `projects/{project_id}/databases/{database_id}/do
      cuments/{document_path}`. This bloom filter may be omitted at the
      server's discretion, such as if it is deemed that the client will not
      make use of it or if it is too computationally expensive to calculate or
      transmit. Clients must gracefully handle this field being absent by
      falling back to the logic used before this field existed; that is, re-
      add the target without a resume token to figure out which documents in
      the client's cache are out of sync.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    targetId = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    unchangedNames = _messages.MessageField('BloomFilter', 3)