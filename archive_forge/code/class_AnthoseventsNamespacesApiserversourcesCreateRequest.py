from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesApiserversourcesCreateRequest(_messages.Message):
    """A AnthoseventsNamespacesApiserversourcesCreateRequest object.

  Fields:
    apiServerSource: A ApiServerSource resource to be passed as the request
      body.
    parent: The project ID or project number in which this apiserversource
      should be created.
  """
    apiServerSource = _messages.MessageField('ApiServerSource', 1)
    parent = _messages.StringField(2, required=True)