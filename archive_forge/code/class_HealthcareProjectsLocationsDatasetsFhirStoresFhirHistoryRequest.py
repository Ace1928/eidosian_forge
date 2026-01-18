from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirHistoryRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirHistoryRequest
  object.

  Fields:
    _at: Only include resource versions that were current at some point during
      the time period specified in the date time value. The date parameter
      format is yyyy-mm-ddThh:mm:ss[Z|(+|-)hh:mm] Clients may specify any of
      the following: * An entire year: `_at=2019` * An entire month:
      `_at=2019-01` * A specific day: `_at=2019-01-20` * A specific second:
      `_at=2018-12-31T23:59:58Z`
    _count: The maximum number of search results on a page. If not specified,
      100 is used. May not be larger than 1000.
    _page_token: Used to retrieve the first, previous, next, or last page of
      resource versions when using pagination. Value should be set to the
      value of `_page_token` set in next or previous page links' URLs. Next
      and previous page are returned in the response bundle's links field,
      where `link.relation` is "previous" or "next". Omit `_page_token` if no
      previous request has been made.
    _since: Only include resource versions that were created at or after the
      given instant in time. The instant in time uses the format YYYY-MM-
      DDThh:mm:ss.sss+zz:zz (for example 2015-02-07T13:28:17.239+02:00 or
      2017-01-01T00:00:00Z). The time must be specified to the second and
      include a time zone.
    name: The name of the resource to retrieve.
  """
    _at = _messages.StringField(1)
    _count = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    _page_token = _messages.StringField(3)
    _since = _messages.StringField(4)
    name = _messages.StringField(5, required=True)