from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditReport(_messages.Message):
    """Represents an audit report.

  Fields:
    complianceStandard: Output only. Compliance Standard.
    controlDetails: Output only. The overall status of controls
    createTime: Output only. Creation time of the audit report.
    destinationDetails: Output only. The location where the generated report
      will be uploaded.
    name: Identifier. The name of this Audit Report, in the format of scope
      given in request.
    operationId: Output only. ClientOperationId
    reportSummary: Output only. Report summary with compliance, violation
      counts etc.
    scope: Output only. The parent scope on which the report was generated.
  """
    complianceStandard = _messages.StringField(1)
    controlDetails = _messages.MessageField('ControlDetails', 2, repeated=True)
    createTime = _messages.StringField(3)
    destinationDetails = _messages.MessageField('DestinationDetails', 4)
    name = _messages.StringField(5)
    operationId = _messages.StringField(6)
    reportSummary = _messages.MessageField('ReportSummary', 7)
    scope = _messages.StringField(8)