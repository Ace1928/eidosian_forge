import grpc
from keras_tuner.src.protos.v4 import (
class OracleStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetSpace = channel.unary_unary('/keras_tuner.Oracle/GetSpace', request_serializer=keras__tuner_dot_protos_dot_service__pb2.GetSpaceRequest.SerializeToString, response_deserializer=keras__tuner_dot_protos_dot_service__pb2.GetSpaceResponse.FromString)
        self.UpdateSpace = channel.unary_unary('/keras_tuner.Oracle/UpdateSpace', request_serializer=keras__tuner_dot_protos_dot_service__pb2.UpdateSpaceRequest.SerializeToString, response_deserializer=keras__tuner_dot_protos_dot_service__pb2.UpdateSpaceResponse.FromString)
        self.CreateTrial = channel.unary_unary('/keras_tuner.Oracle/CreateTrial', request_serializer=keras__tuner_dot_protos_dot_service__pb2.CreateTrialRequest.SerializeToString, response_deserializer=keras__tuner_dot_protos_dot_service__pb2.CreateTrialResponse.FromString)
        self.UpdateTrial = channel.unary_unary('/keras_tuner.Oracle/UpdateTrial', request_serializer=keras__tuner_dot_protos_dot_service__pb2.UpdateTrialRequest.SerializeToString, response_deserializer=keras__tuner_dot_protos_dot_service__pb2.UpdateTrialResponse.FromString)
        self.EndTrial = channel.unary_unary('/keras_tuner.Oracle/EndTrial', request_serializer=keras__tuner_dot_protos_dot_service__pb2.EndTrialRequest.SerializeToString, response_deserializer=keras__tuner_dot_protos_dot_service__pb2.EndTrialResponse.FromString)
        self.GetBestTrials = channel.unary_unary('/keras_tuner.Oracle/GetBestTrials', request_serializer=keras__tuner_dot_protos_dot_service__pb2.GetBestTrialsRequest.SerializeToString, response_deserializer=keras__tuner_dot_protos_dot_service__pb2.GetBestTrialsResponse.FromString)
        self.GetTrial = channel.unary_unary('/keras_tuner.Oracle/GetTrial', request_serializer=keras__tuner_dot_protos_dot_service__pb2.GetTrialRequest.SerializeToString, response_deserializer=keras__tuner_dot_protos_dot_service__pb2.GetTrialResponse.FromString)