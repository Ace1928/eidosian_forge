import numpy as np
from werkzeug import wrappers
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.mesh import metadata
from tensorboard.plugins.mesh import plugin_data_pb2
from tensorboard import plugin_util
def _instance_tags(self, ctx, experiment, run, tag):
    """Gets the instance tag names for a user-facing tag."""
    index = self._data_provider.list_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, run_tag_filter=provider.RunTagFilter(runs=[run]))
    return [instance_tag for instance_tag, ts in index.get(run, {}).items() if tag == metadata.parse_plugin_metadata(ts.plugin_content).name]