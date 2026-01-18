from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class GetSettingsRequest(proto.Message):
    """The parameters to
    [GetSettings][google.logging.v2.ConfigServiceV2.GetSettings].

    See [View default resource settings for Logging]
    (https://cloud.google.com/logging/docs/default-settings#view-org-settings)
    for more information.

    Attributes:
        name (str):
            Required. The resource for which to retrieve settings.

            ::

                "projects/[PROJECT_ID]/settings"
                "organizations/[ORGANIZATION_ID]/settings"
                "billingAccounts/[BILLING_ACCOUNT_ID]/settings"
                "folders/[FOLDER_ID]/settings"

            For example:

            ``"organizations/12345/settings"``

            Note: Settings can be retrieved for Google Cloud projects,
            folders, organizations, and billing accounts.
    """
    name: str = proto.Field(proto.STRING, number=1)