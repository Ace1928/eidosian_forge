from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class UpdateCmekSettingsRequest(proto.Message):
    """The parameters to
    [UpdateCmekSettings][google.logging.v2.ConfigServiceV2.UpdateCmekSettings].

    See `Enabling CMEK for Log
    Router <https://cloud.google.com/logging/docs/routing/managed-encryption>`__
    for more information.

    Attributes:
        name (str):
            Required. The resource name for the CMEK settings to update.

            ::

                "projects/[PROJECT_ID]/cmekSettings"
                "organizations/[ORGANIZATION_ID]/cmekSettings"
                "billingAccounts/[BILLING_ACCOUNT_ID]/cmekSettings"
                "folders/[FOLDER_ID]/cmekSettings"

            For example:

            ``"organizations/12345/cmekSettings"``

            Note: CMEK for the Log Router can currently only be
            configured for Google Cloud organizations. Once configured,
            it applies to all projects and folders in the Google Cloud
            organization.
        cmek_settings (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.CmekSettings):
            Required. The CMEK settings to update.

            See `Enabling CMEK for Log
            Router <https://cloud.google.com/logging/docs/routing/managed-encryption>`__
            for more information.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Optional. Field mask identifying which fields from
            ``cmek_settings`` should be updated. A field will be
            overwritten if and only if it is in the update mask. Output
            only fields cannot be updated.

            See [FieldMask][google.protobuf.FieldMask] for more
            information.

            For example: ``"updateMask=kmsKeyName"``
    """
    name: str = proto.Field(proto.STRING, number=1)
    cmek_settings: 'CmekSettings' = proto.Field(proto.MESSAGE, number=2, message='CmekSettings')
    update_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask)