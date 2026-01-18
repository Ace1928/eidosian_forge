import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.backend import process_graph
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graph_util
from tensorboard.plugins.graph import keras_util
from tensorboard.plugins.graph import metadata
from tensorboard.util import tb_logging
def _read_blob(self, ctx, experiment, plugin_names, run, tag):
    for plugin_name in plugin_names:
        blob_sequences = self._data_provider.read_blob_sequences(ctx, experiment_id=experiment, plugin_name=plugin_name, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]), downsample=1)
        blob_sequence_data = blob_sequences.get(run, {}).get(tag, ())
        try:
            blob_ref = blob_sequence_data[0].values[0]
        except IndexError:
            continue
        return self._data_provider.read_blob(ctx, blob_key=blob_ref.blob_key)
    raise errors.NotFoundError()