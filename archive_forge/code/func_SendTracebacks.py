import grpc
from tensorflow.core.debug import debug_service_pb2 as tensorflow_dot_core_dot_debug_dot_debug__service__pb2
from tensorflow.core.protobuf import debug_pb2 as tensorflow_dot_core_dot_protobuf_dot_debug__pb2
from tensorflow.core.util import event_pb2 as tensorflow_dot_core_dot_util_dot_event__pb2
def SendTracebacks(self, request, context):
    """Send the tracebacks of ops in a Python graph definition.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')