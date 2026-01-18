from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataScansJobsListRequest(_messages.Message):
    """A DataplexProjectsLocationsDataScansJobsListRequest object.

  Fields:
    filter: Optional. An expression for filtering the results of the
      ListDataScanJobs request.If unspecified, all datascan jobs will be
      returned. Multiple filters can be applied (with AND, OR logical
      operators). Filters are case-sensitive.Allowed fields are: start_time
      end_timestart_time and end_time expect RFC-3339 formatted strings (e.g.
      2018-10-08T18:30:00-07:00).For instance, 'start_time >
      2018-10-08T00:00:00.123456789Z AND end_time <
      2018-10-09T00:00:00.123456789Z' limits results to DataScanJobs between
      specified start and end times.
    pageSize: Optional. Maximum number of DataScanJobs to return. The service
      may return fewer than this value. If unspecified, at most 10
      DataScanJobs will be returned. The maximum value is 1000; values above
      1000 will be coerced to 1000.
    pageToken: Optional. Page token received from a previous ListDataScanJobs
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to ListDataScanJobs must match the call that
      provided the page token.
    parent: Required. The resource name of the parent environment:
      projects/{project}/locations/{location_id}/dataScans/{data_scan_id}
      where project refers to a project_id or project_number and location_id
      refers to a GCP region.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)