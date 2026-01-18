from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpOrganizationsLocationsProjectDataProfilesListRequest(_messages.Message):
    """A DlpOrganizationsLocationsProjectDataProfilesListRequest object.

  Fields:
    filter: Allows filtering. Supported syntax: * Filter expressions are made
      up of one or more restrictions. * Restrictions can be combined by `AND`
      or `OR` logical operators. A sequence of restrictions implicitly uses
      `AND`. * A restriction has the form of `{field} {operator} {value}`. *
      Supported fields/values: - `sensitivity_level` - HIGH|MODERATE|LOW -
      `data_risk_level` - HIGH|MODERATE|LOW - `status_code` - an RPC status
      code as defined in https://github.com/googleapis/googleapis/blob/master/
      google/rpc/code.proto * The operator must be `=` or `!=`. Examples: *
      `project_id = 12345 AND status_code = 1` * `project_id = 12345 AND
      sensitivity_level = HIGH` The length of this field should be no more
      than 500 characters.
    orderBy: Comma separated list of fields to order by, followed by `asc` or
      `desc` postfix. This list is case insensitive. The default sorting order
      is ascending. Redundant space characters are insignificant. Only one
      order field at a time is allowed. Examples: * `project_id` *
      `sensitivity_level desc` Supported fields are: - `project_id`: Google
      Cloud project ID - `sensitivity_level`: How sensitive the data in a
      project is, at most. - `data_risk_level`: How much risk is associated
      with this data. - `profile_last_generated`: When the profile was last
      updated in epoch seconds.
    pageSize: Size of the page. This value can be limited by the server. If
      zero, server returns a page of max size 100.
    pageToken: Page token to continue retrieval.
    parent: Required. organizations/{org_id}/locations/{loc_id}
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)