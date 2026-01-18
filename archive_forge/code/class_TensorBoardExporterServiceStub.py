import grpc
from tensorboard.uploader.proto import export_service_pb2 as tensorboard_dot_uploader_dot_proto_dot_export__service__pb2
class TensorBoardExporterServiceStub(object):
    """Service for exporting data from TensorBoard.dev.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.StreamExperiments = channel.unary_stream('/tensorboard.service.TensorBoardExporterService/StreamExperiments', request_serializer=tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamExperimentsRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamExperimentsResponse.FromString)
        self.StreamExperimentData = channel.unary_stream('/tensorboard.service.TensorBoardExporterService/StreamExperimentData', request_serializer=tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamExperimentDataRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamExperimentDataResponse.FromString)
        self.StreamBlobData = channel.unary_stream('/tensorboard.service.TensorBoardExporterService/StreamBlobData', request_serializer=tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamBlobDataRequest.SerializeToString, response_deserializer=tensorboard_dot_uploader_dot_proto_dot_export__service__pb2.StreamBlobDataResponse.FromString)