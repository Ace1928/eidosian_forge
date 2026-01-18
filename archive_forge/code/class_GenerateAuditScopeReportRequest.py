from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateAuditScopeReportRequest(_messages.Message):
    """Message for requesting audit scope report.

  Enums:
    ReportFormatValueValuesEnum: Required. The format in which the Scope
      report bytes should be returned.

  Fields:
    complianceStandard: Required. Compliance Standard against which the Scope
      Report must be generated. Eg: FEDRAMP_MODERATE
    reportFormat: Required. The format in which the Scope report bytes should
      be returned.
  """

    class ReportFormatValueValuesEnum(_messages.Enum):
        """Required. The format in which the Scope report bytes should be
    returned.

    Values:
      AUDIT_SCOPE_REPORT_FORMAT_UNSPECIFIED: Unspecified. Invalid format.
      AUDIT_SCOPE_REPORT_FORMAT_ODF: Audit Scope Report creation format is
        Open Document.
    """
        AUDIT_SCOPE_REPORT_FORMAT_UNSPECIFIED = 0
        AUDIT_SCOPE_REPORT_FORMAT_ODF = 1
    complianceStandard = _messages.StringField(1)
    reportFormat = _messages.EnumField('ReportFormatValueValuesEnum', 2)