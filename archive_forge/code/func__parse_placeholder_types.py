import argparse
import os
import sys
from absl import app
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib
def _parse_placeholder_types(values):
    """Extracts placeholder types from a comma separate list."""
    values = [int(value) for value in values.split(',')]
    return values if len(values) > 1 else values[0]