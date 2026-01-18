from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetSavedQueriesDeleteRequest(_messages.Message):
    """A CloudassetSavedQueriesDeleteRequest object.

  Fields:
    name: Required. The name of the saved query to delete. It must be in the
      format of: * projects/project_number/savedQueries/saved_query_id *
      folders/folder_number/savedQueries/saved_query_id *
      organizations/organization_number/savedQueries/saved_query_id
  """
    name = _messages.StringField(1, required=True)