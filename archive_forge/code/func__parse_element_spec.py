import multiprocessing
import os
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.data.experimental.service import _pywrap_snapshot_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import nested_structure_coder
def _parse_element_spec(encoded_element_spec):
    struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
    struct_pb.ParseFromString(encoded_element_spec)
    return nested_structure_coder.decode_proto(struct_pb)