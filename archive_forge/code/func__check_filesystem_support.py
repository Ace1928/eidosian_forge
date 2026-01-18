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
def _check_filesystem_support(paths):
    """Examines the list of filesystems user requested.

    If TF I/O schemes are requested, try to import tensorflow_io module.

    Args:
        paths: A list of strings representing input log directories.
    """
    get_registered_schemes = getattr(tf.io.gfile, 'get_registered_schemes', None)
    registered_schemes = None if get_registered_schemes is None else get_registered_schemes()
    scheme_to_path = {_get_filesystem_scheme(path): path for path in paths}
    missing_scheme = None
    for scheme, path in scheme_to_path.items():
        if scheme is None:
            continue
        if registered_schemes is not None:
            if scheme not in registered_schemes:
                missing_scheme = scheme
                break
        else:
            try:
                tf.io.gfile.exists(path)
            except tf.errors.UnimplementedError:
                missing_scheme = scheme
                break
            except tf.errors.OpError:
                pass
    if missing_scheme:
        try:
            import tensorflow_io
        except ImportError as e:
            supported_schemes_msg = ' (supported schemes: {})'.format(registered_schemes) if registered_schemes else ''
            raise tf.errors.UnimplementedError(None, None, ("Error: Unsupported filename scheme '{}'{}. For additional" + ' filesystem support, consider installing TensorFlow I/O' + ' (https://www.tensorflow.org/io) via `pip install tensorflow-io`.').format(missing_scheme, supported_schemes_msg)) from e