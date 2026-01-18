from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemotebuildexecutionProjectsInstancesWorkerpoolsListRequest(_messages.Message):
    """A RemotebuildexecutionProjectsInstancesWorkerpoolsListRequest object.

  Fields:
    filter: Optional. A filter expression that filters resources listed in the
      response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. String values are case-
      insensitive. The comparison operator must be either `:`, `=`, `!=`, `>`,
      `>=`, `<=` or `<`. The `:` operator can be used with string fields to
      match substrings. For non-string fields it is equivalent to the `=`
      operator. The `:*` comparison can be used to test whether a key has been
      defined. You can also filter on nested fields. To filter on multiple
      expressions, you can separate expression using `AND` and `OR` operators,
      using parentheses to specify precedence. If neither operator is
      specified, `AND` is assumed. Examples: Include only pools with more than
      100 reserved workers: `(worker_count > 100) (worker_config.reserved =
      true)` Include only pools with a certain label or machines of the
      e2-standard family: `worker_config.labels.key1 : * OR
      worker_config.machine_type: e2-standard`
    parent: Resource name of the instance. Format:
      `projects/[PROJECT_ID]/instances/[INSTANCE_ID]`.
  """
    filter = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)