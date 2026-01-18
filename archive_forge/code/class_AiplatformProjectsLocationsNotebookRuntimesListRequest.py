from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimesListRequest object.

  Fields:
    filter: Optional. An expression for filtering the results of the request.
      For field names both snake_case and camelCase are supported. *
      `notebookRuntime` supports = and !=. `notebookRuntime` represents the
      NotebookRuntime ID, i.e. the last segment of the NotebookRuntime's
      resource name. * `displayName` supports = and != and regex. *
      `notebookRuntimeTemplate` supports = and !=. `notebookRuntimeTemplate`
      represents the NotebookRuntimeTemplate ID, i.e. the last segment of the
      NotebookRuntimeTemplate's resource name. * `healthState` supports = and
      !=. healthState enum: [HEALTHY, UNHEALTHY, HEALTH_STATE_UNSPECIFIED]. *
      `runtimeState` supports = and !=. runtimeState enum:
      [RUNTIME_STATE_UNSPECIFIED, RUNNING, BEING_STARTED, BEING_STOPPED,
      STOPPED, BEING_UPGRADED, ERROR, INVALID]. * `runtimeUser` supports = and
      !=. * API version is UI only: `uiState` supports = and !=. uiState enum:
      [UI_RESOURCE_STATE_UNSPECIFIED, UI_RESOURCE_STATE_BEING_CREATED,
      UI_RESOURCE_STATE_ACTIVE, UI_RESOURCE_STATE_BEING_DELETED,
      UI_RESOURCE_STATE_CREATION_FAILED]. * `notebookRuntimeType` supports =
      and !=. notebookRuntimeType enum: [USER_DEFINED, ONE_CLICK]. Some
      examples: * `notebookRuntime="notebookRuntime123"` *
      `displayName="myDisplayName"` and `displayName=~"myDisplayNameRegex"` *
      `notebookRuntimeTemplate="notebookRuntimeTemplate321"` *
      `healthState=HEALTHY` * `runtimeState=RUNNING` *
      `runtimeUser="test@google.com"` *
      `uiState=UI_RESOURCE_STATE_BEING_DELETED` *
      `notebookRuntimeType=USER_DEFINED`
    orderBy: Optional. A comma-separated list of fields to order by, sorted in
      ascending order. Use "desc" after a field name for descending. Supported
      fields: * `display_name` * `create_time` * `update_time` Example:
      `display_name, create_time desc`.
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token. Typically obtained via
      ListNotebookRuntimesResponse.next_page_token of the previous
      NotebookService.ListNotebookRuntimes call.
    parent: Required. The resource name of the Location from which to list the
      NotebookRuntimes. Format: `projects/{project}/locations/{location}`
    readMask: Optional. Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)