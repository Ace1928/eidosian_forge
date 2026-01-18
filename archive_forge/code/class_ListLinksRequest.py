from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class ListLinksRequest(proto.Message):
    """The parameters to ListLinks.

    Attributes:
        parent (str):
            Required. The parent resource whose links are to be listed:

            ::

                "projects/[PROJECT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"
                "organizations/[ORGANIZATION_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]"
                "folders/[FOLDER_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]".
        page_token (str):
            Optional. If present, then retrieve the next batch of
            results from the preceding call to this method.
            ``pageToken`` must be the value of ``nextPageToken`` from
            the previous response.
        page_size (int):
            Optional. The maximum number of results to
            return from this request.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_token: str = proto.Field(proto.STRING, number=2)
    page_size: int = proto.Field(proto.INT32, number=3)