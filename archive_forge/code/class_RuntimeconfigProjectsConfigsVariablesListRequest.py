from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsVariablesListRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsVariablesListRequest object.

  Fields:
    filter: Filters variables by matching the specified filter. For example:
      `projects/example-project/config/[CONFIG_NAME]/variables/example-
      variable`.
    pageSize: Specifies the number of results to return per page. If there are
      fewer elements than the specified number, returns all elements.
    pageToken: Specifies a page token to use. Set `pageToken` to a
      `nextPageToken` returned by a previous list request to get the next page
      of results.
    parent: The path to the RuntimeConfig resource for which you want to list
      variables. The configuration must exist beforehand; the path must be in
      the format: `projects/[PROJECT_ID]/configs/[CONFIG_NAME]`
    returnValues: The flag indicates whether the user wants to return values
      of variables. If true, then only those variables that user has IAM
      GetVariable permission will be returned along with their values.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    returnValues = _messages.BooleanField(5)