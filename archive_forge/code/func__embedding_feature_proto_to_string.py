import enum
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _embedding_feature_proto_to_string(cls, embedding_feature_proto):
    """Convert the embedding feature proto to enum string."""
    embedding_feature_proto_to_string_map = {topology_pb2.TPUHardwareFeature.EmbeddingFeature.UNSUPPORTED: HardwareFeature.EmbeddingFeature.UNSUPPORTED, topology_pb2.TPUHardwareFeature.EmbeddingFeature.V1: HardwareFeature.EmbeddingFeature.V1, topology_pb2.TPUHardwareFeature.EmbeddingFeature.V2: HardwareFeature.EmbeddingFeature.V2}
    return embedding_feature_proto_to_string_map.get(embedding_feature_proto, HardwareFeature.EmbeddingFeature.UNSUPPORTED)