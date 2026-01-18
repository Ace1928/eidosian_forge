import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
def _handle_tpu_embedding(self, tpu_embedding):
    """Handle TPUEmbedding.

    Args:
      tpu_embedding: TPUEmbedding object to be handled.

    Raises:
      AttributeError: if the input trackable is not TPUEmbedding type.
    """
    if not hasattr(tpu_embedding, _TPU_EMBEDDING_ATTR) or not callable(tpu_embedding._create_copy_for_async_checkpoint):
        raise AttributeError('Expecting TPUEmbedding type; got %s' % type(tpu_embedding))
    new_embedding = tpu_embedding._create_copy_for_async_checkpoint(feature_config=tpu_embedding._feature_config, optimizer=tpu_embedding._table_config[0] if tpu_embedding._table_config else None, pipeline_execution_with_tensor_core=tpu_embedding._pipeline_execution_with_tensor_core)
    self._object_map[tpu_embedding] = new_embedding
    if tpu_embedding not in self._tpu_embedding_objects:
        self._tpu_embedding_objects.append(tpu_embedding)