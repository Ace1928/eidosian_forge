from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class DeleteSavedQueryRequest(proto.Message):
    """The parameters to 'DeleteSavedQuery'.

    Attributes:
        name (str):
            Required. The full resource name of the saved query to
            delete.

            ::

                "projects/[PROJECT_ID]/locations/[LOCATION_ID]/savedQueries/[QUERY_ID]"
                "organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/savedQueries/[QUERY_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/savedQueries/[QUERY_ID]"
                "folders/[FOLDER_ID]/locations/[LOCATION_ID]/savedQueries/[QUERY_ID]"

            For example:

            ::

                "projects/my-project/locations/global/savedQueries/my-saved-query".
    """
    name: str = proto.Field(proto.STRING, number=1)