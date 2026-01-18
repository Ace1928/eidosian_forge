from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ResponseMessageKnowledgeInfoCard(_messages.Message):
    """Represents info card response. If the response contains generative
  knowledge prediction, Dialogflow will return a payload with Infobot
  Messenger compatible info card. Otherwise, the info card response is
  skipped.
  """