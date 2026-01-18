from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorDetailsValueListEntry(_messages.Message):
    """A ErrorDetailsValueListEntry object.

        Fields:
          errorInfo: A ErrorInfo attribute.
          help: A Help attribute.
          localizedMessage: A LocalizedMessage attribute.
          quotaInfo: A QuotaExceededInfo attribute.
        """
    errorInfo = _messages.MessageField('ErrorInfo', 1)
    help = _messages.MessageField('Help', 2)
    localizedMessage = _messages.MessageField('LocalizedMessage', 3)
    quotaInfo = _messages.MessageField('QuotaExceededInfo', 4)