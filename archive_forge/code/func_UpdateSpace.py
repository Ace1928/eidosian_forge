import grpc
from keras_tuner.src.protos.v4 import (
@staticmethod
def UpdateSpace(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
    return grpc.experimental.unary_unary(request, target, '/keras_tuner.Oracle/UpdateSpace', keras__tuner_dot_protos_dot_service__pb2.UpdateSpaceRequest.SerializeToString, keras__tuner_dot_protos_dot_service__pb2.UpdateSpaceResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)