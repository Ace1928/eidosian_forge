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
def _fix_string_types(self, ndarray):
    """Change the dtype of text arrays to String rather than Object.

        np.savez ends up pickling np.object arrays, while it doesn't pickle
        strings.  The downside is that it needs to pad the length of each string
        in the array to the maximal length string.  We only want to do this
        type override in this final step of the export path.

        Args:
          ndarray: a tensor converted to an ndarray

        Returns:
          The original ndarray if not np.object, dtype converted to String
          if np.object.
        """
    if ndarray.dtype != np.object_:
        return ndarray
    else:
        return ndarray.astype('|S')