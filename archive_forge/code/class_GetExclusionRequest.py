from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class GetExclusionRequest(proto.Message):
    """The parameters to ``GetExclusion``.

    Attributes:
        name (str):
            Required. The resource name of an existing exclusion:

            ::

                "projects/[PROJECT_ID]/exclusions/[EXCLUSION_ID]"
                "organizations/[ORGANIZATION_ID]/exclusions/[EXCLUSION_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/exclusions/[EXCLUSION_ID]"
                "folders/[FOLDER_ID]/exclusions/[EXCLUSION_ID]"

            For example:

            ``"projects/my-project/exclusions/my-exclusion"``
    """
    name: str = proto.Field(proto.STRING, number=1)