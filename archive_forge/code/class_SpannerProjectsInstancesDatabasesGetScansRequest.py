from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesGetScansRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesGetScansRequest object.

  Enums:
    ViewValueValuesEnum: Specifies which parts of the Scan should be returned
      in the response. Note, if left unspecified, the FULL view is assumed.

  Fields:
    endTime: The upper bound for the time range to retrieve Scan data for.
    name: Required. The unique name of the scan containing the requested
      information, specific to the Database service implementing this
      interface.
    startTime: These fields restrict the Database-specific information
      returned in the `Scan.data` field. If a `View` is provided that does not
      include the `Scan.data` field, these are ignored. This range of time
      must be entirely contained within the defined time range of the targeted
      scan. The lower bound for the time range to retrieve Scan data for.
    view: Specifies which parts of the Scan should be returned in the
      response. Note, if left unspecified, the FULL view is assumed.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies which parts of the Scan should be returned in the response.
    Note, if left unspecified, the FULL view is assumed.

    Values:
      VIEW_UNSPECIFIED: Not specified, equivalent to SUMMARY.
      SUMMARY: Server responses only include `name`, `details`, `start_time`
        and `end_time`. The default value. Note, the ListScans method may only
        use this view type, others view types are not supported.
      FULL: Full representation of the scan is returned in the server
        response, including `data`.
    """
        VIEW_UNSPECIFIED = 0
        SUMMARY = 1
        FULL = 2
    endTime = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    startTime = _messages.StringField(3)
    view = _messages.EnumField('ViewValueValuesEnum', 4)