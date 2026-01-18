from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigMapKeySelector(_messages.Message):
    """Not supported by Cloud Run.

  Fields:
    key: Required. Not supported by Cloud Run.
    localObjectReference: Not supported by Cloud Run.
    name: Required. Not supported by Cloud Run.
    optional: Not supported by Cloud Run.
  """
    key = _messages.StringField(1)
    localObjectReference = _messages.MessageField('LocalObjectReference', 2)
    name = _messages.StringField(3)
    optional = _messages.BooleanField(4)