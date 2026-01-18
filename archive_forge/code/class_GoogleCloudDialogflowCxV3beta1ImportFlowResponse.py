from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ImportFlowResponse(_messages.Message):
    """The response message for Flows.ImportFlow.

  Fields:
    flow: The unique identifier of the new flow. Format:
      `projects//locations//agents//flows/`.
  """
    flow = _messages.StringField(1)