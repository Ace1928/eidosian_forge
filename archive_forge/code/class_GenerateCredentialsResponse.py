from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GenerateCredentialsResponse(_messages.Message):
    """Connection information for a particular membership.

  Fields:
    endpoint: The generated URI of the cluster as accessed through the Connect
      Gateway API.
    kubeconfig: A full YAML kubeconfig in serialized format.
  """
    endpoint = _messages.StringField(1)
    kubeconfig = _messages.BytesField(2)