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
def _get_tag_run_image_info(mapping):
    """Returns a map of tag names to run information.

    Args:
        mapping: the result of DataProvider's `list_blob_sequences`.

    Returns:
        A nested map from run strings to tag string to image info, where image
        info is an object of form {"maxSamplesPerStep": num}. For example,
        {
            "reshaped": {
                "test": {"maxSamplesPerStep": 1},
                "train": {"maxSamplesPerStep": 1}
            },
            "convolved": {"test": {"maxSamplesPerStep": 50}},
        }
    """
    tag_run_image_info = collections.defaultdict(dict)
    for run, tag_to_content in mapping.items():
        for tag, metadatum in tag_to_content.items():
            tag_run_image_info[tag][run] = {'maxSamplesPerStep': metadatum.max_length - 2}
    return dict(tag_run_image_info)