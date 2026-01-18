from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryDomainAliasesListRequest(_messages.Message):
    """A DirectoryDomainAliasesListRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
    parentDomainName: Name of the parent domain for which domain aliases are
      to be fetched.
  """
    customer = _messages.StringField(1, required=True)
    parentDomainName = _messages.StringField(2)