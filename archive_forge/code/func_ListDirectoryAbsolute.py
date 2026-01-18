import collections
import os
import re
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def ListDirectoryAbsolute(directory):
    """Yields all files in the given directory.

    The paths are absolute.
    """
    return (os.path.join(directory, path) for path in tf.io.gfile.listdir(directory))