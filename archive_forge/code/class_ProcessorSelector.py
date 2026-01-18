import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class ProcessorSelector(proto.Message):
    """-

        Attributes:
            processor_names (Sequence[str]):
                -
            processor (str):
                -
            device_config_key ((google.cloud.quantum_v1alpha1.types.DeviceConfigKey):
                -
        """
    processor_names = proto.RepeatedField(proto.STRING, number=1)
    processor = proto.Field(proto.STRING, number=2)
    device_config_key = proto.Field(proto.MESSAGE, number=3, message=DeviceConfigKey)