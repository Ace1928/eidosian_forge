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
def _get_tag_to_description(mapping):
    """Returns a map of tags to descriptions.

    Args:
        mapping: a nested map `d` such that `d[run][tag]` is a time series
          produced by DataProvider's `list_*` methods.

    Returns:
        A map from tag strings to description HTML strings. E.g.
        {
            "loss": "<h1>Multiple descriptions</h1><h2>For runs: test, train
            </h2><p>...</p>",
            "loss2": "<p>The lossy details</p>",
        }
    """
    tag_to_descriptions, description_to_runs = _get_tag_description_info(mapping)
    result = {}
    for tag in tag_to_descriptions:
        descriptions = sorted(tag_to_descriptions[tag])
        if len(descriptions) == 1:
            description = descriptions[0]
        else:
            description = _build_combined_description(descriptions, description_to_runs)
        result[tag] = plugin_util.markdown_to_safe_html(description)
    return result