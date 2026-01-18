import grpc
from tensorboard.uploader.proto import write_service_pb2 as tensorboard_dot_uploader_dot_proto_dot_write__service__pb2
@staticmethod
def WriteBlob(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.stream_stream(request_iterator, target, '/tensorboard.service.TensorBoardWriterService/WriteBlob', tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteBlobRequest.SerializeToString, tensorboard_dot_uploader_dot_proto_dot_write__service__pb2.WriteBlobResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)