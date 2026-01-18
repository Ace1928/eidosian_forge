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
def _process_blob_sequence_points(self, blob_sequences, experiment_id):
    """Process blob sequence points.

        As a side effect, also downloads the binary contents of the blobs
        to respective files. The paths to the files relative to the
        experiment directory is encapsulated in the returned JSON object.

        Args:
          blob_sequences:
            `export_service_pb2.StreamDataResponse.BlobSequencePoints` proto.

        Returns:
          A JSON-serializable `dict` for the steps and wall_times, as well as
            the blob_file_paths, which are the relative paths to the downloaded
            blob contents.
        """
    wall_times = [t.ToNanoseconds() / 1000000000.0 for t in blob_sequences.wall_times]
    json_object = {'steps': list(blob_sequences.steps), 'wall_times': wall_times, 'blob_file_paths': []}
    blob_file_paths = json_object['blob_file_paths']
    for blobseq in blob_sequences.values:
        seq_blob_file_paths = []
        for entry in blobseq.entries:
            if entry.blob.state == blob_pb2.BlobState.BLOB_STATE_CURRENT:
                blob_path = self._download_blob(entry.blob.blob_id, experiment_id)
                seq_blob_file_paths.append(blob_path)
            else:
                seq_blob_file_paths.append(None)
        blob_file_paths.append(seq_blob_file_paths)
    return json_object