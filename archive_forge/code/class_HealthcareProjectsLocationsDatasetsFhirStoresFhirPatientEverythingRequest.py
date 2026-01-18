from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirPatientEverythingRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsFhirStoresFhirPatientEverythingRequest
  object.

  Fields:
    _count: Maximum number of resources in a page. If not specified, 100 is
      used. May not be larger than 1000.
    _page_token: Used to retrieve the next or previous page of results when
      using pagination. Set `_page_token` to the value of _page_token set in
      next or previous page links' url. Next and previous page are returned in
      the response bundle's links field, where `link.relation` is "previous"
      or "next". Omit `_page_token` if no previous request has been made.
    _since: If provided, only resources updated after this time are returned.
      The time uses the format YYYY-MM-DDThh:mm:ss.sss+zz:zz. For example,
      `2015-02-07T13:28:17.239+02:00` or `2017-01-01T00:00:00Z`. The time must
      be specified to the second and include a time zone.
    _type: String of comma-delimited FHIR resource types. If provided, only
      resources of the specified resource type(s) are returned. Specifying
      multiple `_type` parameters isn't supported. For example, the result of
      `_type=Observation&_type=Encounter` is undefined. Use
      `_type=Observation,Encounter` instead.
    end: The response includes records prior to the end date. The date uses
      the format YYYY-MM-DD. If no end date is provided, all records
      subsequent to the start date are in scope.
    name: Name of the `Patient` resource for which the information is
      required.
    start: The response includes records subsequent to the start date. The
      date uses the format YYYY-MM-DD. If no start date is provided, all
      records prior to the end date are in scope.
  """
    _count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    _page_token = _messages.StringField(2)
    _since = _messages.StringField(3)
    _type = _messages.StringField(4)
    end = _messages.StringField(5)
    name = _messages.StringField(6, required=True)
    start = _messages.StringField(7)