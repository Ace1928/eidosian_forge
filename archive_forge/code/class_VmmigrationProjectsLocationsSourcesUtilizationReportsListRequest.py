from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesUtilizationReportsListRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesUtilizationReportsListRequest
  object.

  Enums:
    ViewValueValuesEnum: Optional. The level of details of each report.
      Defaults to BASIC.

  Fields:
    filter: Optional. The filter request.
    orderBy: Optional. the order by fields for the result.
    pageSize: Optional. The maximum number of reports to return. The service
      may return fewer than this value. If unspecified, at most 500 reports
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: Required. A page token, received from a previous
      `ListUtilizationReports` call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to
      `ListUtilizationReports` must match the call that provided the page
      token.
    parent: Required. The Utilization Reports parent.
    view: Optional. The level of details of each report. Defaults to BASIC.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. The level of details of each report. Defaults to BASIC.

    Values:
      UTILIZATION_REPORT_VIEW_UNSPECIFIED: The default / unset value. The API
        will default to FULL on single report request and BASIC for multiple
        reports request.
      BASIC: Get the report metadata, without the list of VMs and their
        utilization info.
      FULL: Include everything.
    """
        UTILIZATION_REPORT_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 6)