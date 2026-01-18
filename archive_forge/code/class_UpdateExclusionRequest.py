from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class UpdateExclusionRequest(proto.Message):
    """The parameters to ``UpdateExclusion``.

    Attributes:
        name (str):
            Required. The resource name of the exclusion to update:

            ::

                "projects/[PROJECT_ID]/exclusions/[EXCLUSION_ID]"
                "organizations/[ORGANIZATION_ID]/exclusions/[EXCLUSION_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/exclusions/[EXCLUSION_ID]"
                "folders/[FOLDER_ID]/exclusions/[EXCLUSION_ID]"

            For example:

            ``"projects/my-project/exclusions/my-exclusion"``
        exclusion (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.LogExclusion):
            Required. New values for the existing exclusion. Only the
            fields specified in ``update_mask`` are relevant.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. A non-empty list of fields to change in the
            existing exclusion. New values for the fields are taken from
            the corresponding fields in the
            [LogExclusion][google.logging.v2.LogExclusion] included in
            this request. Fields not mentioned in ``update_mask`` are
            not changed and are ignored in the request.

            For example, to change the filter and description of an
            exclusion, specify an ``update_mask`` of
            ``"filter,description"``.
    """
    name: str = proto.Field(proto.STRING, number=1)
    exclusion: 'LogExclusion' = proto.Field(proto.MESSAGE, number=2, message='LogExclusion')
    update_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask)