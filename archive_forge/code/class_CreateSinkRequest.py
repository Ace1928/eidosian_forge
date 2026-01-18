from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class CreateSinkRequest(proto.Message):
    """The parameters to ``CreateSink``.

    Attributes:
        parent (str):
            Required. The resource in which to create the sink:

            ::

                "projects/[PROJECT_ID]"
                "organizations/[ORGANIZATION_ID]"
                "billingAccounts/[BILLING_ACCOUNT_ID]"
                "folders/[FOLDER_ID]"

            For examples:

            ``"projects/my-project"`` ``"organizations/123456789"``
        sink (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.LogSink):
            Required. The new sink, whose ``name`` parameter is a sink
            identifier that is not already in use.
        unique_writer_identity (bool):
            Optional. Determines the kind of IAM identity returned as
            ``writer_identity`` in the new sink. If this value is
            omitted or set to false, and if the sink's parent is a
            project, then the value returned as ``writer_identity`` is
            the same group or service account used by Cloud Logging
            before the addition of writer identities to this API. The
            sink's destination must be in the same project as the sink
            itself.

            If this field is set to true, or if the sink is owned by a
            non-project resource such as an organization, then the value
            of ``writer_identity`` will be a `service
            agent <https://cloud.google.com/iam/docs/service-account-types#service-agents>`__
            used by the sinks with the same parent. For more
            information, see ``writer_identity`` in
            [LogSink][google.logging.v2.LogSink].
        custom_writer_identity (str):
            Optional. A service account provided by the caller that will
            be used to write the log entries. The format must be
            ``serviceAccount:some@email``. This field can only be
            specified if you are routing logs to a destination outside
            this sink's project. If not specified, a Logging service
            account will automatically be generated.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    sink: 'LogSink' = proto.Field(proto.MESSAGE, number=2, message='LogSink')
    unique_writer_identity: bool = proto.Field(proto.BOOL, number=3)
    custom_writer_identity: str = proto.Field(proto.STRING, number=4)