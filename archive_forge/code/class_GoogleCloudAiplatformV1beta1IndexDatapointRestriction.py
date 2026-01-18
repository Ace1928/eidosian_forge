from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1IndexDatapointRestriction(_messages.Message):
    """Restriction of a datapoint which describe its attributes(tokens) from
  each of several attribute categories(namespaces).

  Fields:
    allowList: The attributes to allow in this namespace. e.g.: 'red'
    denyList: The attributes to deny in this namespace. e.g.: 'blue'
    namespace: The namespace of this restriction. e.g.: color.
  """
    allowList = _messages.StringField(1, repeated=True)
    denyList = _messages.StringField(2, repeated=True)
    namespace = _messages.StringField(3)