from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryDomainAliasesInsertRequest(_messages.Message):
    """A DirectoryDomainAliasesInsertRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
    domainAlias: A DomainAlias resource to be passed as the request body.
  """
    customer = _messages.StringField(1, required=True)
    domainAlias = _messages.MessageField('DomainAlias', 2)