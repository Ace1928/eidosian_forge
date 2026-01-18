from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2WordList(_messages.Message):
    """Message defining a list of words or phrases to search for in the data.

  Fields:
    words: Words or phrases defining the dictionary. The dictionary must
      contain at least one phrase and every phrase must contain at least 2
      characters that are letters or digits. [required]
  """
    words = _messages.StringField(1, repeated=True)