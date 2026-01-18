import contextlib
import functools
import time
import grpc
from google.protobuf import message
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.uploader.proto import write_service_pb2
from tensorboard.uploader import logdir_loader
from tensorboard.uploader import upload_tracker
from tensorboard.uploader import util
from tensorboard.backend import process_graph
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _get_or_create_blob_sequence(self):
    request = write_service_pb2.GetOrCreateBlobSequenceRequest(experiment_id=self._experiment_id, run=self._run_name, tag=self._value.tag, step=self._event.step, final_sequence_length=len(self._blobs), metadata=self._metadata)
    util.set_timestamp(request.wall_time, self._event.wall_time)
    with _request_logger(request):
        try:
            response = grpc_util.call_with_retries(self._api.GetOrCreateBlobSequence, request)
            blob_sequence_id = response.blob_sequence_id
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise ExperimentNotFoundError()
            logger.error('Upload call failed with error %s', e)
            raise
    return blob_sequence_id