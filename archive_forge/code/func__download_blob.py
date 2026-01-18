import base64
import contextlib
import errno
import grpc
import json
import os
import string
import time
import numpy as np
from tensorboard.uploader.proto import blob_pb2
from tensorboard.uploader.proto import experiment_pb2
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader import util
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _download_blob(self, blob_id, experiment_id):
    """Download the blob via rpc.

        Args:
          blob_id: Id of the blob.
          experiment_id: Id of the experiment that the blob belongs to.

        Returns:
          If the blob is downloaded successfully:
            The path of the downloaded blob file relative to the experiment
            directory.
          Else:
            `None`.
        """
    experiment_dir = _experiment_directory(self._outdir, experiment_id)
    request = export_service_pb2.StreamBlobDataRequest(blob_id=blob_id)
    blob_abspath = os.path.join(experiment_dir, _DIRNAME_BLOBS, _FILENAME_BLOBS_PREFIX + blob_id + _FILENAME_BLOBS_SUFFIX)
    with open(blob_abspath, 'xb') as f:
        try:
            for response in self._api.StreamBlobData(request, metadata=grpc_util.version_metadata()):
                f.write(response.data)
        except grpc.RpcError as rpc_error:
            logger.error('Omitting blob (id: %s) due to download failure: %s', blob_id, rpc_error)
            return None
    return os.path.relpath(blob_abspath, experiment_dir)