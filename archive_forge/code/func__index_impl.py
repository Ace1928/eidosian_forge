import imghdr
import urllib.parse
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata
def _index_impl(self, ctx, experiment):
    mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME)
    result = {run: {} for run in mapping}
    for run, tag_to_content in mapping.items():
        for tag, metadatum in tag_to_content.items():
            md = metadata.parse_plugin_metadata(metadatum.plugin_content)
            if not self._version_checker.ok(md.version, run, tag):
                continue
            description = plugin_util.markdown_to_safe_html(metadatum.description)
            result[run][tag] = {'displayName': metadatum.display_name, 'description': description, 'samples': metadatum.max_length - 2}
    return result