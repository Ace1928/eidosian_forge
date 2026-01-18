from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ABNFGrammar(_messages.Message):
    """A ABNFGrammar object.

  Fields:
    abnfStrings: All declarations and rules of an ABNF grammar broken up into
      multiple strings that will end up concatenated.
  """
    abnfStrings = _messages.StringField(1, repeated=True)