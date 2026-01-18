from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsConversationsListRequest(_messages.Message):
    """A DialogflowProjectsConversationsListRequest object.

  Fields:
    filter: A filter expression that filters conversations listed in the
      response. In general, the expression must specify the field name, a
      comparison operator, and the value to use for filtering: - The value
      must be a string, a number, or a boolean. - The comparison operator must
      be either `=`,`!=`, `>`, or `<`. - To filter on multiple expressions,
      separate the expressions with `AND` or `OR` (omitting both implies
      `AND`). - For clarity, expressions can be enclosed in parentheses. Only
      `lifecycle_state` can be filtered on in this way. For example, the
      following expression only returns `COMPLETED` conversations:
      `lifecycle_state = "COMPLETED"` For more information about filtering,
      see [API Filtering](https://aip.dev/160).
    pageSize: Optional. The maximum number of items to return in a single
      page. By default 100 and at most 1000.
    pageToken: Optional. The next_page_token value returned from a previous
      list request.
    parent: Required. The project from which to list all conversation. Format:
      `projects//locations/`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)