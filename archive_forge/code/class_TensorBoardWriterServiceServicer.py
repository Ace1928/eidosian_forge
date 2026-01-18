import grpc
from tensorboard.uploader.proto import write_service_pb2 as tensorboard_dot_uploader_dot_proto_dot_write__service__pb2
class TensorBoardWriterServiceServicer(object):
    """Service for writing data to TensorBoard.dev.
    """

    def CreateExperiment(self, request, context):
        """Request for a new location to write TensorBoard readable events.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateExperiment(self, request, context):
        """Request to mutate metadata associated with an experiment.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteExperiment(self, request, context):
        """Request that an experiment be deleted, along with all tags and scalars
        that it contains. This call may only be made by the original owner of the
        experiment.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PurgeData(self, request, context):
        """Request that unreachable data be purged. Used only for testing;
        disabled in production.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def WriteScalar(self, request, context):
        """Request additional scalar data be stored in TensorBoard.dev.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def WriteTensor(self, request, context):
        """Request additional tensor data be stored in TensorBoard.dev.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetOrCreateBlobSequence(self, request, context):
        """Request to obtain a specific BlobSequence entry, creating it if needed,
        to be subsequently populated with blobs.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetBlobMetadata(self, request, context):
        """Request the current status of blob data being stored in TensorBoard.dev,
        to support resumable uploads.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def WriteBlob(self, request_iterator, context):
        """Request additional blob data be stored in TensorBoard.dev.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteOwnUser(self, request, context):
        """Request that the calling user and all their data be permanently deleted.
        Used for testing purposes.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')