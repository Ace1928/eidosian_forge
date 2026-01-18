from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResponsePolicyRuleLocalData(_messages.Message):
    """A ResponsePolicyRuleLocalData object.

  Fields:
    localDatas: All resource record sets for this selector, one per resource
      record type. The name must match the dns_name.
  """
    localDatas = _messages.MessageField('ResourceRecordSet', 1, repeated=True)