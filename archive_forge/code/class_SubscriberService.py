import grpc
from . import pubsub_pb2 as src_dot_ray_dot_protobuf_dot_pubsub__pb2
class SubscriberService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def PubsubLongPolling(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.SubscriberService/PubsubLongPolling', src_dot_ray_dot_protobuf_dot_pubsub__pb2.PubsubLongPollingRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_pubsub__pb2.PubsubLongPollingReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PubsubCommandBatch(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.SubscriberService/PubsubCommandBatch', src_dot_ray_dot_protobuf_dot_pubsub__pb2.PubsubCommandBatchRequest.SerializeToString, src_dot_ray_dot_protobuf_dot_pubsub__pb2.PubsubCommandBatchReply.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)