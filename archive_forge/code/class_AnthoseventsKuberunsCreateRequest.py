from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsKuberunsCreateRequest(_messages.Message):
    """A AnthoseventsKuberunsCreateRequest object.

  Fields:
    kubeRun: A KubeRun resource to be passed as the request body.
    parent: The namespace in which this KubeRun resource should be created.
  """
    kubeRun = _messages.MessageField('KubeRun', 1)
    parent = _messages.StringField(2)