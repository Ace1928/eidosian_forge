from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2EntityId(_messages.Message):
    """An entity in a dataset is a field or set of fields that correspond to a
  single person. For example, in medical records the `EntityId` might be a
  patient identifier, or for financial records it might be an account
  identifier. This message is used when generalizations or analysis must take
  into account that multiple rows correspond to the same entity.

  Fields:
    field: Composite key indicating which field contains the entity
      identifier.
  """
    field = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1)