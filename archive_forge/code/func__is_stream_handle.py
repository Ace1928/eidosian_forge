import io
import json
import os.path
import pickle
import tempfile
import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper
@staticmethod
def _is_stream_handle(data):
    obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
    return isinstance(obj_to_check, (io.BufferedIOBase, io.RawIOBase))