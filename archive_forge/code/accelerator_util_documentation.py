from typing import List, Optional
from absl import logging
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.platform import remote_utils
from tensorflow.python.util.tf_export import tf_export
Shuts down the accelerator system.