from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingFoldersLocationsSavedQueriesCreateRequest(_messages.Message):
    """A LoggingFoldersLocationsSavedQueriesCreateRequest object.

  Fields:
    parent: Required. The parent resource in which to create the saved query:
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]"
      "organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]"
      "folders/[FOLDER_ID]/locations/[LOCATION_ID]" For example: "projects/my-
      project/locations/global" "organizations/123456789/locations/us-
      central1"
    savedQuery: A SavedQuery resource to be passed as the request body.
    savedQueryId: Optional. The ID to use for the saved query, which will
      become the final component of the saved query's resource name.If the
      saved_query_id is not provided, the system will generate an alphanumeric
      ID.The saved_query_id is limited to 100 characters and can include only
      the following characters: upper and lower-case alphanumeric characters,
      underscores, hyphens, periods.First character has to be alphanumeric.
  """
    parent = _messages.StringField(1, required=True)
    savedQuery = _messages.MessageField('SavedQuery', 2)
    savedQueryId = _messages.StringField(3)