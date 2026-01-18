import grpc
from tensorboard.uploader.proto import write_service_pb2 as tensorboard_dot_uploader_dot_proto_dot_write__service__pb2
@staticmethod
def WriteTensor(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/tensorboard.service.TensorBoardWriterService/WriteTensor', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteTensorRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteTensorResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)