from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityActionFlag(_messages.Message):
    """The message that should be set in the case of a Flag action.

  Fields:
    headers: Optional. A list of HTTP headers to be sent to the target in case
      of a FLAG SecurityAction. Limit 5 headers per SecurityAction. At least
      one is mandatory.
  """
    headers = _messages.MessageField('GoogleCloudApigeeV1SecurityActionHttpHeader', 1, repeated=True)