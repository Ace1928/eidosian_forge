from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsSourcesLocationsFindingsListRequest(_messages.Message):
    """A SecuritycenterProjectsSourcesLocationsFindingsListRequest object.

  Fields:
    fieldMask: A field mask to specify the Finding fields to be listed in the
      response. An empty field mask will list all fields.
    filter: Expression that defines the filter to apply across findings. The
      expression is a list of one or more restrictions combined via logical
      operators `AND` and `OR`. Parentheses are supported, and `OR` has higher
      precedence than `AND`. Restrictions have the form ` ` and may have a `-`
      character in front of them to indicate negation. Examples include: *
      name * security_marks.marks.marka The supported operators are: * `=` for
      all value types. * `>`, `<`, `>=`, `<=` for integer values. * `:`,
      meaning substring matching, for strings. The supported value types are:
      * string literals in quotes. * integer literals without quotes. *
      boolean literals `true` and `false` without quotes. The following field
      and operator combinations are supported: * name: `=` * parent: `=`, `:`
      * resource_name: `=`, `:` * state: `=`, `:` * category: `=`, `:` *
      external_uri: `=`, `:` * event_time: `=`, `>`, `<`, `>=`, `<=` Usage:
      This should be milliseconds since epoch or an RFC3339 string. Examples:
      `event_time = "2019-06-10T16:07:18-07:00"` `event_time = 1560208038000`
      * severity: `=`, `:` * security_marks.marks: `=`, `:` * resource: *
      resource.name: `=`, `:` * resource.parent_name: `=`, `:` *
      resource.parent_display_name: `=`, `:` * resource.project_name: `=`, `:`
      * resource.project_display_name: `=`, `:` * resource.type: `=`, `:` *
      resource.folders.resource_folder: `=`, `:` * resource.display_name: `=`,
      `:`
    orderBy: Expression that defines what fields and order to use for sorting.
      The string value should follow SQL syntax: comma separated list of
      fields. For example: "name,parent". The default sorting order is
      ascending. To specify descending order for a field, a suffix " desc"
      should be appended to the field name. For example: "name desc,parent".
      Redundant space characters in the syntax are insignificant. "name
      desc,parent" and " name desc , parent " are equivalent. The following
      fields are supported: name parent state category resource_name
      event_time security_marks.marks
    pageSize: The maximum number of results to return in a single response.
      Default is 10, minimum is 1, maximum is 1000.
    pageToken: The value returned by the last `ListFindingsResponse`;
      indicates that this is a continuation of a prior `ListFindings` call,
      and that the system should return the next page of data.
    parent: Required. Name of the source the findings belong to. If no
      location is specified, the default is global. The following list shows
      some examples: + `organizations/[organization_id]/sources/[source_id]` +
      `organizations/[organization_id]/sources/[source_id]/locations/[location
      _id]` + `folders/[folder_id]/sources/[source_id]` +
      `folders/[folder_id]/sources/[source_id]/locations/[location_id]` +
      `projects/[project_id]/sources/[source_id]` +
      `projects/[project_id]/sources/[source_id]/locations/[location_id]` To
      list across all sources provide a source_id of `-`. The following list
      shows some examples: + `organizations/{organization_id}/sources/-` +
      `organizations/{organization_id}/sources/-/locations/{location_id}` +
      `folders/{folder_id}/sources/-` +
      `folders/{folder_id}/sources/-locations/{location_id}` +
      `projects/{projects_id}/sources/-` +
      `projects/{projects_id}/sources/-/locations/{location_id}`
  """
    fieldMask = _messages.StringField(1)
    filter = _messages.StringField(2)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    parent = _messages.StringField(6, required=True)