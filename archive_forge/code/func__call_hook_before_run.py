import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
def _call_hook_before_run(self, run_context, fetch_dict, user_feed_dict, options):
    """Calls hooks.before_run and handles requests from hooks."""
    hook_feeds = {}
    for hook in self._hooks:
        request = hook.before_run(run_context)
        if request is not None:
            if request.fetches is not None:
                fetch_dict[hook] = request.fetches
            if request.feed_dict:
                self._raise_if_feeds_intersects(hook_feeds, request.feed_dict, 'Same tensor is fed by two hooks.')
                hook_feeds.update(request.feed_dict)
            if request.options:
                self._merge_run_options(options, request.options)
    if not hook_feeds:
        return user_feed_dict
    if not user_feed_dict:
        return hook_feeds
    self._raise_if_feeds_intersects(user_feed_dict, hook_feeds, 'Same tensor is fed by a SessionRunHook and user.')
    hook_feeds.update(user_feed_dict)
    return hook_feeds