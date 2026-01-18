import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _encode_blob_key(experiment_id, plugin_name, run, tag, step, index):
    """Generate a blob key: a short, URL-safe string identifying a blob.

    A blob can be located using a set of integer and string fields; here we
    serialize these to allow passing the data through a URL.  Specifically, we
    1) construct a tuple of the arguments in order; 2) represent that as an
    ascii-encoded JSON string (without whitespace); and 3) take the URL-safe
    base64 encoding of that, with no padding.  For example:

        1)  Tuple: ("some_id", "graphs", "train", "graph_def", 2, 0)
        2)   JSON: ["some_id","graphs","train","graph_def",2,0]
        3) base64: WyJzb21lX2lkIiwiZ3JhcGhzIiwidHJhaW4iLCJncmFwaF9kZWYiLDIsMF0K

    Args:
      experiment_id: a string ID identifying an experiment.
      plugin_name: string
      run: string
      tag: string
      step: int
      index: int

    Returns:
      A URL-safe base64-encoded string representing the provided arguments.
    """
    stringified = json.dumps((experiment_id, plugin_name, run, tag, step, index), separators=(',', ':'))
    bytesified = stringified.encode('ascii')
    encoded = base64.urlsafe_b64encode(bytesified)
    return encoded.decode('ascii').rstrip('=')