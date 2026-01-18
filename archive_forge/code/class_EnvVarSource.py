from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnvVarSource(_messages.Message):
    """EnvVarSource represents a source for the value of an EnvVar.

  Fields:
    configMapKeyRef: Not supported by Cloud Run. Not supported in Cloud Run.
    secretKeyRef: Selects a key (version) of a secret in Secret Manager.
  """
    configMapKeyRef = _messages.MessageField('ConfigMapKeySelector', 1)
    secretKeyRef = _messages.MessageField('SecretKeySelector', 2)