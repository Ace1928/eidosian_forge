from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class CreateExclusionRequest(proto.Message):
    """The parameters to ``CreateExclusion``.

    Attributes:
        parent (str):
            Required. The parent resource in which to create the
            exclusion:

            ::

                "projects/[PROJECT_ID]"
                "organizations/[ORGANIZATION_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]"
                "folders/[FOLDER_ID]"

            For examples:

            ``"projects/my-logging-project"``
            ``"organizations/123456789"``
        exclusion (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.LogExclusion):
            Required. The new exclusion, whose ``name`` parameter is an
            exclusion name that is not already used in the parent
            resource.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    exclusion: 'LogExclusion' = proto.Field(proto.MESSAGE, number=2, message='LogExclusion')