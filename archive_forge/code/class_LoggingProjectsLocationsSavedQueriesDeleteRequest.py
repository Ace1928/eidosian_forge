from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingProjectsLocationsSavedQueriesDeleteRequest(_messages.Message):
    """A LoggingProjectsLocationsSavedQueriesDeleteRequest object.

  Fields:
    name: Required. The full resource name of the saved query to delete.
      "projects/[PROJECT_ID]/locations/[LOCATION_ID]/savedQueries/[QUERY_ID]"
      "organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/savedQueries/[Q
      UERY_ID]" "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/
      savedQueries/[QUERY_ID]"
      "folders/[FOLDER_ID]/locations/[LOCATION_ID]/savedQueries/[QUERY_ID]"
      For example: "projects/my-project/locations/global/savedQueries/my-
      saved-query"
  """
    name = _messages.StringField(1, required=True)