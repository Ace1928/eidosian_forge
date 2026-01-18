from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2KAnonymityConfig(_messages.Message):
    """k-anonymity metric, used for analysis of reidentification risk.

  Fields:
    entityId: Message indicating that multiple rows might be associated to a
      single individual. If the same entity_id is associated to multiple
      quasi-identifier tuples over distinct rows, we consider the entire
      collection of tuples as the composite quasi-identifier. This collection
      is a multiset: the order in which the different tuples appear in the
      dataset is ignored, but their frequency is taken into account. Important
      note: a maximum of 1000 rows can be associated to a single entity ID. If
      more rows are associated with the same entity ID, some might be ignored.
    quasiIds: Set of fields to compute k-anonymity over. When multiple fields
      are specified, they are considered a single composite key. Structs and
      repeated data types are not supported; however, nested fields are
      supported so long as they are not structs themselves or nested within a
      repeated field.
  """
    entityId = _messages.MessageField('GooglePrivacyDlpV2EntityId', 1)
    quasiIds = _messages.MessageField('GooglePrivacyDlpV2FieldId', 2, repeated=True)