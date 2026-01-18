import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _convert_blob_sequence_event(experiment_id, plugin_name, run, tag, event):
    """Helper for `read_blob_sequences`."""
    num_blobs = _tensor_size(event.tensor_proto)
    values = tuple((provider.BlobReference(_encode_blob_key(experiment_id, plugin_name, run, tag, event.step, idx)) for idx in range(num_blobs)))
    return provider.BlobSequenceDatum(wall_time=event.wall_time, step=event.step, values=values)