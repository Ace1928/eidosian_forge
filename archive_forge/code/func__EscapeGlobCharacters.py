import collections
import os
import re
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def _EscapeGlobCharacters(path):
    """Escapes the glob characters in a path.

    Python 3 has a glob.escape method, but python 2 lacks it, so we manually
    implement this method.

    Args:
      path: The absolute path to escape.

    Returns:
      The escaped path string.
    """
    drive, path = os.path.splitdrive(path)
    return '%s%s' % (drive, _ESCAPE_GLOB_CHARACTERS_REGEX.sub('[\\1]', path))