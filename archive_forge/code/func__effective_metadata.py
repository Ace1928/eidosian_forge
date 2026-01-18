import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
def _effective_metadata(metadata, metadata_transformer):
    non_none_metadata = () if metadata is None else metadata
    if metadata_transformer is None:
        return non_none_metadata
    else:
        return metadata_transformer(non_none_metadata)