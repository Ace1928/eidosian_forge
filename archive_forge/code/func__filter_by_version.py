import collections
import imghdr
import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.metrics import metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _filter_by_version(self, mapping, parse_metadata, version_checker):
    """Filter `DataProvider.list_*` output by summary metadata version."""
    result = {run: {} for run in mapping}
    for run, tag_to_content in mapping.items():
        for tag, metadatum in tag_to_content.items():
            md = parse_metadata(metadatum.plugin_content)
            if not version_checker.ok(md.version, run, tag):
                continue
            result[run][tag] = metadatum
    return result