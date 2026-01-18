from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsGlobalConnectivityTestsListRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsGlobalConnectivityTestsListRequest
  object.

  Fields:
    filter: Lists the `ConnectivityTests` that match the filter expression. A
      filter expression filters the resources listed in the response. The
      expression must be of the form ` ` where operators: `<`, `>`, `<=`,
      `>=`, `!=`, `=`, `:` are supported (colon `:` represents a HAS operator
      which is roughly synonymous with equality). can refer to a proto or JSON
      field, or a synthetic field. Field names can be camelCase or snake_case.
      Examples: - Filter by name: name =
      "projects/proj-1/locations/global/connectivityTests/test-1 - Filter by
      labels: - Resources that have a key called `foo` labels.foo:* -
      Resources that have a key called `foo` whose value is `bar` labels.foo =
      bar
    orderBy: Field to use to sort the list.
    pageSize: Number of `ConnectivityTests` to return.
    pageToken: Page token from an earlier query, as returned in
      `next_page_token`.
    parent: Required. The parent resource of the Connectivity Tests:
      `projects/{project_id}/locations/global`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)