from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsStoredInfoTypesListRequest(_messages.Message):
    """A DlpProjectsStoredInfoTypesListRequest object.

  Fields:
    locationId: Deprecated. This field has no effect.
    orderBy: Comma separated list of fields to order by, followed by `asc` or
      `desc` postfix. This list is case insensitive. The default sorting order
      is ascending. Redundant space characters are insignificant. Example:
      `name asc, display_name, create_time desc` Supported fields are: -
      `create_time`: corresponds to the time the most recent version of the
      resource was created. - `state`: corresponds to the state of the
      resource. - `name`: corresponds to resource name. - `display_name`:
      corresponds to info type's display name.
    pageSize: Size of the page. This value can be limited by the server. If
      zero server returns a page of max size 100.
    pageToken: Page token to continue retrieval. Comes from the previous call
      to `ListStoredInfoTypes`.
    parent: Required. Parent resource name. The format of this value varies
      depending on the scope of the request (project or organization) and
      whether you have [specified a processing
      location](https://cloud.google.com/sensitive-data-
      protection/docs/specifying-location): + Projects scope, location
      specified: `projects/`PROJECT_ID`/locations/`LOCATION_ID + Projects
      scope, no location specified (defaults to global): `projects/`PROJECT_ID
      The following example `parent` string specifies a parent project with
      the identifier `example-project`, and specifies the `europe-west3`
      location for processing data: parent=projects/example-
      project/locations/europe-west3
  """
    locationId = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)