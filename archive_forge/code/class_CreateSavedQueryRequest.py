from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class CreateSavedQueryRequest(proto.Message):
    """The parameters to 'CreateSavedQuery'.

    Attributes:
        parent (str):
            Required. The parent resource in which to create the saved
            query:

            ::

                "projects/[PROJECT_ID]/locations/[LOCATION_ID]"
                "organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]"
                "folders/[FOLDER_ID]/locations/[LOCATION_ID]"

            For example:

            ::

                "projects/my-project/locations/global"
                "organizations/123456789/locations/us-central1".
        saved_query_id (str):
            Optional. The ID to use for the saved query, which will
            become the final component of the saved query's resource
            name.

            If the ``saved_query_id`` is not provided, the system will
            generate an alphanumeric ID.

            The ``saved_query_id`` is limited to 100 characters and can
            include only the following characters:

            -  upper and lower-case alphanumeric characters,
            -  underscores,
            -  hyphens,
            -  periods.

            First character has to be alphanumeric.
        saved_query (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.SavedQuery):
            Required. The new saved query.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    saved_query_id: str = proto.Field(proto.STRING, number=3)
    saved_query: 'SavedQuery' = proto.Field(proto.MESSAGE, number=2, message='SavedQuery')