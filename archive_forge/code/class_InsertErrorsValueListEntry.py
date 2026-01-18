from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InsertErrorsValueListEntry(_messages.Message):
    """A InsertErrorsValueListEntry object.

    Fields:
      errors: Error information for the row indicated by the index property.
      index: The index of the row that error applies to.
    """
    errors = _messages.MessageField('ErrorProto', 1, repeated=True)
    index = _messages.IntegerField(2, variant=_messages.Variant.UINT32)