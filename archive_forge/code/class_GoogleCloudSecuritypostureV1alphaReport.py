from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritypostureV1alphaReport(_messages.Message):
    """========================== Reports ========================== Definition
  of the resource 'Report'.

  Fields:
    createTime: Output only. The timestamp when the report was created.
    iacValidationReport: A IaCValidationReport attribute.
    name: Required. The name of this Report resource, in the format of
      organizations/{organization}/locations/{location}/reports/{reportID}.
    updateTime: Output only. The timestamp when the report was updated.
  """
    createTime = _messages.StringField(1)
    iacValidationReport = _messages.MessageField('IaCValidationReport', 2)
    name = _messages.StringField(3)
    updateTime = _messages.StringField(4)