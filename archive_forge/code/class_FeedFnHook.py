import os
import time
import numpy as np
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.FeedFnHook'])
class FeedFnHook(session_run_hook.SessionRunHook):
    """Runs `feed_fn` and sets the `feed_dict` accordingly."""

    def __init__(self, feed_fn):
        """Initializes a `FeedFnHook`.

    Args:
      feed_fn: function that takes no arguments and returns `dict` of `Tensor`
        to feed.
    """
        self.feed_fn = feed_fn

    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs(fetches=None, feed_dict=self.feed_fn())