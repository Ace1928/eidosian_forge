import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class QuantumJob(proto.Message):
    """-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            -
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        labels (Sequence[google.cloud.quantum_v1alpha1.types.QuantumJob.LabelsEntry]):
            -
        label_fingerprint (str):
            -
        description (str):
            -
        scheduling_config (google.cloud.quantum_v1alpha1.types.SchedulingConfig):
            -
        output_config (google.cloud.quantum_v1alpha1.types.OutputConfig):
            -
        execution_status (google.cloud.quantum_v1alpha1.types.ExecutionStatus):
            -
        gcs_run_context_location (google.cloud.quantum_v1alpha1.types.GcsLocation):
            -

            This field is a member of `oneof`_ ``run_context_location``.
        run_context_inline_data (google.cloud.quantum_v1alpha1.types.InlineData):
            -

            This field is a member of `oneof`_ ``run_context_location``.
        run_context (google.protobuf.any_pb2.Any):
            -
    """
    name = proto.Field(proto.STRING, number=1)
    create_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    update_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    labels = proto.MapField(proto.STRING, proto.STRING, number=4)
    label_fingerprint = proto.Field(proto.STRING, number=5)
    description = proto.Field(proto.STRING, number=6)
    scheduling_config = proto.Field(proto.MESSAGE, number=7, message='SchedulingConfig')
    output_config = proto.Field(proto.MESSAGE, number=8, message='OutputConfig')
    execution_status = proto.Field(proto.MESSAGE, number=9, message='ExecutionStatus')
    gcs_run_context_location = proto.Field(proto.MESSAGE, number=10, oneof='run_context_location', message='GcsLocation')
    run_context_inline_data = proto.Field(proto.MESSAGE, number=12, oneof='run_context_location', message='InlineData')
    run_context = proto.Field(proto.MESSAGE, number=11, message=any_pb2.Any)