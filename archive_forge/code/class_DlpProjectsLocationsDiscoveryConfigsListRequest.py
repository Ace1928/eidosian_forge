from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsDiscoveryConfigsListRequest(_messages.Message):
    """A DlpProjectsLocationsDiscoveryConfigsListRequest object.

  Fields:
    orderBy: Comma separated list of config fields to order by, followed by
      `asc` or `desc` postfix. This list is case insensitive. The default
      sorting order is ascending. Redundant space characters are
      insignificant. Example: `name asc,update_time, create_time desc`
      Supported fields are: - `last_run_time`: corresponds to the last time
      the DiscoveryConfig ran. - `name`: corresponds to the DiscoveryConfig's
      name. - `status`: corresponds to DiscoveryConfig's status.
    pageSize: Size of the page. This value can be limited by a server.
    pageToken: Page token to continue retrieval. Comes from the previous call
      to ListDiscoveryConfigs. `order_by` field must not change for subsequent
      calls.
    parent: Required. Parent resource name. The format of this value is as
      follows: `projects/`PROJECT_ID`/locations/`LOCATION_ID The following
      example `parent` string specifies a parent project with the identifier
      `example-project`, and specifies the `europe-west3` location for
      processing data: parent=projects/example-project/locations/europe-west3
  """
    orderBy = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)