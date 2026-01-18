import collections
import copy
import csv
import json
import os
import re
import sys
import time
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options as checkpoint_options_lib
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _configure_embeddings(self):
    """Configure the Projector for embeddings."""
    from google.protobuf import text_format
    from tensorflow.python.keras.layers import embeddings
    from tensorflow.python.keras.protobuf import projector_config_pb2
    config = projector_config_pb2.ProjectorConfig()
    for layer in self.model.layers:
        if isinstance(layer, embeddings.Embedding):
            embedding = config.embeddings.add()
            name = 'layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE'
            embedding.tensor_name = name
            if self.embeddings_metadata is not None:
                if isinstance(self.embeddings_metadata, str):
                    embedding.metadata_path = self.embeddings_metadata
                elif layer.name in self.embeddings_metadata.keys():
                    embedding.metadata_path = self.embeddings_metadata.pop(layer.name)
    if self.embeddings_metadata and (not isinstance(self.embeddings_metadata, str)):
        raise ValueError('Unrecognized `Embedding` layer names passed to `keras.callbacks.TensorBoard` `embeddings_metadata` argument: ' + str(self.embeddings_metadata.keys()))
    config_pbtxt = text_format.MessageToString(config)
    path = os.path.join(self._log_write_dir, 'projector_config.pbtxt')
    with gfile.Open(path, 'w') as f:
        f.write(config_pbtxt)