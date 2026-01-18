import collections
import dataclasses
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import event_util
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.event_processing import reservoir
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import meta_graph_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.util import tb_logging
def PluginTagToContent(self, plugin_name):
    """Returns a dict mapping tags to content specific to that plugin.

        Args:
          plugin_name: The name of the plugin for which to fetch plugin-specific
            content.

        Raises:
          KeyError: if the plugin name is not found.

        Returns:
          A dict mapping tag names to bytestrings of plugin-specific content-- by
          convention, in the form of binary serialized protos.
        """
    with self._plugin_tag_lock:
        if plugin_name not in self._plugin_to_tag_to_content:
            raise KeyError('Plugin %r could not be found.' % plugin_name)
        return dict(self._plugin_to_tag_to_content[plugin_name])