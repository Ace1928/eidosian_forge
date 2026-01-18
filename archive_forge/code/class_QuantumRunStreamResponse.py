import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class QuantumRunStreamResponse(proto.Message):
    """-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        message_id (str):
            -
        error (google.cloud.quantum_v1alpha1.types.StreamError):
            -

            This field is a member of `oneof`_ ``response``.
        job (google.cloud.quantum_v1alpha1.types.QuantumJob):
            -

            This field is a member of `oneof`_ ``response``.
        result (google.cloud.quantum_v1alpha1.types.QuantumResult):
            -

            This field is a member of `oneof`_ ``response``.
    """
    message_id = proto.Field(proto.STRING, number=1)
    error = proto.Field(proto.MESSAGE, number=2, oneof='response', message='StreamError')
    job = proto.Field(proto.MESSAGE, number=3, oneof='response', message=quantum.QuantumJob)
    result = proto.Field(proto.MESSAGE, number=4, oneof='response', message=quantum.QuantumResult)