import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
class _DynamicStub(face.DynamicStub):

    def __init__(self, backing_generic_stub, group, cardinalities):
        self._generic_stub = backing_generic_stub
        self._group = group
        self._cardinalities = cardinalities

    def __getattr__(self, attr):
        method_cardinality = self._cardinalities.get(attr)
        if method_cardinality is cardinality.Cardinality.UNARY_UNARY:
            return self._generic_stub.unary_unary(self._group, attr)
        elif method_cardinality is cardinality.Cardinality.UNARY_STREAM:
            return self._generic_stub.unary_stream(self._group, attr)
        elif method_cardinality is cardinality.Cardinality.STREAM_UNARY:
            return self._generic_stub.stream_unary(self._group, attr)
        elif method_cardinality is cardinality.Cardinality.STREAM_STREAM:
            return self._generic_stub.stream_stream(self._group, attr)
        else:
            raise AttributeError('_DynamicStub object has no attribute "%s"!' % attr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False