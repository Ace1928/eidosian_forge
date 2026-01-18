import os
import re
import threading
import time
from tensorboard.backend.event_processing import data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat import tf
from tensorboard.data import ingester
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.pr_curve import metadata as pr_curve_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tb_logging
def _get_filesystem_scheme(path):
    """Extracts filesystem scheme from a given path.

    The filesystem scheme is usually separated by `://` from the local filesystem
    path if given. For example, the scheme of `file://tmp/tf` is `file`.

    Args:
        path: A strings representing an input log directory.
    Returns:
        Filesystem scheme, None if the path doesn't contain one.
    """
    if '://' not in path:
        return None
    return path.split('://')[0]