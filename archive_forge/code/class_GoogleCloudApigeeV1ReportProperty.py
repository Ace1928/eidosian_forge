from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ReportProperty(_messages.Message):
    """A GoogleCloudApigeeV1ReportProperty object.

  Fields:
    property: name of the property
    value: property values
  """
    property = _messages.StringField(1)
    value = _messages.MessageField('GoogleCloudApigeeV1Attribute', 2, repeated=True)