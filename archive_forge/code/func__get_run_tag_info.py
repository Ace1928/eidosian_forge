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
def _get_run_tag_info(mapping):
    """Returns a map of run names to a list of tag names.

    Args:
        mapping: a nested map `d` such that `d[run][tag]` is a time series
          produced by DataProvider's `list_*` methods.

    Returns:
        A map from run strings to a list of tag strings. E.g.
            {"loss001a": ["actor/loss", "critic/loss"], ...}
    """
    return {run: sorted(mapping[run]) for run in mapping}