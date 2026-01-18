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
def _process_tensor_points(self, points, experiment_id):
    """Process tensor data points.

        Args:
          points: `export_service_pb2.StreamExperimentDataResponse.TensorPoints`
            proto.
          experiment_id: ID of the experiment that the `TensorPoints` is a part
            of.

        Returns:
          A JSON-serializable `dict` for the steps, wall_times and the path to
            the .npz files that contain the saved tensor values.
        """
    wall_times = [t.ToNanoseconds() / 1000000000.0 for t in points.wall_times]
    json_object = {'steps': list(points.steps), 'wall_times': wall_times, 'tensors_file_path': None}
    if not json_object['steps']:
        return json_object
    experiment_dir = _experiment_directory(self._outdir, experiment_id)
    tensors_file_path = self._get_tensor_file_path(experiment_dir, json_object['wall_times'][0])
    ndarrays = [tensor_util.make_ndarray(tensor_proto) for tensor_proto in points.values]
    ndarrays = [self._fix_string_types(x) for x in ndarrays]
    np.savez(os.path.join(experiment_dir, tensors_file_path), *ndarrays)
    json_object['tensors_file_path'] = tensors_file_path
    return json_object