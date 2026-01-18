from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class ListSinksRequest(proto.Message):
    """The parameters to ``ListSinks``.

    Attributes:
        parent (str):
            Required. The parent resource whose sinks are to be listed:

            ::

                "projects/[PROJECT_ID]"
                "organizations/[ORGANIZATION_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]"
                "folders/[FOLDER_ID]".
        page_token (str):
            Optional. If present, then retrieve the next batch of
            results from the preceding call to this method.
            ``pageToken`` must be the value of ``nextPageToken`` from
            the previous response. The values of other method parameters
            should be identical to those in the previous call.
        page_size (int):
            Optional. The maximum number of results to return from this
            request. Non-positive values are ignored. The presence of
            ``nextPageToken`` in the response indicates that more
            results might be available.
        filter (str):
            Optional. A filter expression to constrain the sinks
            returned. Today, this only supports the following strings:

            -  ``''``
            -  ``'in_scope("ALL")'``,
            -  ``'in_scope("ANCESTOR")'``,
            -  ``'in_scope("DEFAULT")'``.

            Description of scopes below. ALL: Includes all of the sinks
            which can be returned in any other scope. ANCESTOR: Includes
            intercepting sinks owned by ancestor resources. DEFAULT:
            Includes sinks owned by ``parent``.

            When the empty string is provided, then the filter
            'in_scope("DEFAULT")' is applied.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_token: str = proto.Field(proto.STRING, number=2)
    page_size: int = proto.Field(proto.INT32, number=3)
    filter: str = proto.Field(proto.STRING, number=5)