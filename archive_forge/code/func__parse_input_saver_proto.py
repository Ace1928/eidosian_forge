import argparse
import re
import sys
from absl import app
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as saver_lib
def _parse_input_saver_proto(input_saver, input_binary):
    """Parses input tensorflow Saver into SaverDef proto."""
    if not gfile.Exists(input_saver):
        raise IOError("Input saver file '" + input_saver + "' does not exist!")
    mode = 'rb' if input_binary else 'r'
    with gfile.GFile(input_saver, mode) as f:
        saver_def = saver_pb2.SaverDef()
        if input_binary:
            saver_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), saver_def)
    return saver_def