import collections
import os
import re
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def PathSeparator(path):
    return '/' if io_util.IsCloudPath(path) else os.sep