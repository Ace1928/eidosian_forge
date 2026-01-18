from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupFindingsRequest(_messages.Message):
    """Request message for grouping by findings.

  Fields:
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
      * resource.project_display_name: `=`, `:` * resource.type: `=`, `:`
    groupBy: Required. Expression that defines what assets fields to use for
      grouping. The string value should follow SQL syntax: comma separated
      list of fields. For example: "parent,resource_name". The following
      fields are supported: * resource_name * category * state * parent *
      severity
    pageSize: The maximum number of results to return in a single response.
      Default is 10, minimum is 1, maximum is 1000.
    pageToken: The value returned by the last `GroupFindingsResponse`;
      indicates that this is a continuation of a prior `GroupFindings` call,
      and that the system should return the next page of data.
  """
    filter = _messages.StringField(1)
    groupBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)