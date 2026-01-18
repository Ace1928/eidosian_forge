import grpc
from tensorboard.uploader.proto import export_service_pb2 as tensorboard_dot_uploader_dot_proto_dot_export__service__pb2
class TensorBoardExporterServiceServicer(object):
    """Service for exporting data from TensorBoard.dev.
    """

    def StreamExperiments(self, request, context):
        """Stream the experiment_id of all the experiments owned by the caller.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamExperimentData(self, request, context):
        """Stream scalars for all the runs and tags in an experiment.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamBlobData(self, request, context):
        """Stream blob as chunks for a given blob_id.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')