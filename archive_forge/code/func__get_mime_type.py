import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.audio import metadata
def _get_mime_type(self, ctx, experiment, run, tag):
    mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME)
    time_series = mapping.get(run, {}).get(tag, None)
    if time_series is None:
        raise errors.NotFoundError('No audio data for run=%r, tag=%r' % (run, tag))
    parsed = metadata.parse_plugin_metadata(time_series.plugin_content)
    return _MIME_TYPES.get(parsed.encoding, _DEFAULT_MIME_TYPE)