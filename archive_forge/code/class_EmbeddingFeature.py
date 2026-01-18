import enum
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export
class EmbeddingFeature(enum.Enum):
    """Embedding feature flag strings.

    UNSUPPORTED: No embedding lookup accelerator available on the tpu.
    V1: Embedding lookup accelerator V1. The embedding lookup operation can only
        be placed at the beginning of computation. Only one instance of
        embedding
        lookup layer is allowed.
    V2: Embedding lookup accelerator V2. The embedding lookup operation can be
        placed anywhere of the computation. Multiple instances of embedding
        lookup layer is allowed.
    """
    UNSUPPORTED = 'UNSUPPORTED'
    V1 = 'V1'
    V2 = 'V2'