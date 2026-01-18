from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OndemandscanningProjectsLocationsScansVulnerabilitiesListRequest(_messages.Message):
    """A OndemandscanningProjectsLocationsScansVulnerabilitiesListRequest
  object.

  Fields:
    pageSize: The number of vulnerabilities to retrieve.
    pageToken: The page token, resulting from a previous call to
      ListVulnerabilities.
    parent: Required. The parent of the collection of Vulnerabilities being
      requested. Format:
      projects/[project_name]/locations/[location]/scans/[scan_id]
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)