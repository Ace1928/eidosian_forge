from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1TokensInfo(_messages.Message):
    """Tokens info with a list of tokens and the corresponding list of token
  ids.

  Fields:
    tokenIds: A list of token ids from the input.
    tokens: A list of tokens from the input.
  """
    tokenIds = _messages.IntegerField(1, repeated=True)
    tokens = _messages.BytesField(2, repeated=True)