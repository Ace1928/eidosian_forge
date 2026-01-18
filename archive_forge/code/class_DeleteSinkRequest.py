from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class DeleteSinkRequest(proto.Message):
    """The parameters to ``DeleteSink``.

    Attributes:
        sink_name (str):
            Required. The full resource name of the sink to delete,
            including the parent resource and the sink identifier:

            ::

                "projects/[PROJECT_ID]/sinks/[SINK_ID]"
                "organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]"
                "folders/[FOLDER_ID]/sinks/[SINK_ID]"

            For example:

            ``"projects/my-project/sinks/my-sink"``
    """
    sink_name: str = proto.Field(proto.STRING, number=1)