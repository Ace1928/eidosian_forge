import grpc
from tensorboard.data.proto import data_provider_pb2 as tensorboard_dot_data_dot_proto_dot_data__provider__pb2
class TensorBoardDataProvider(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetExperiment(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/GetExperiment', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.GetExperimentRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.GetExperimentResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListPlugins(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ListPlugins', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListPluginsRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListPluginsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListRuns(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ListRuns', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListRunsRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListRunsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListScalars(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ListScalars', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListScalarsRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListScalarsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReadScalars(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ReadScalars', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadScalarsRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadScalarsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListTensors(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ListTensors', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListTensorsRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListTensorsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReadTensors(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ReadTensors', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadTensorsRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadTensorsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListBlobSequences(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ListBlobSequences', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListBlobSequencesRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ListBlobSequencesResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReadBlobSequences(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorboard.data.TensorBoardDataProvider/ReadBlobSequences', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobSequencesRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobSequencesResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReadBlob(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tensorboard.data.TensorBoardDataProvider/ReadBlob', tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobRequest.SerializeToString, tensorboard_dot_data_dot_proto_dot_data__provider__pb2.ReadBlobResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)