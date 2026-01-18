import os.path
from ... import urlutils
from ...trace import mutter
Determine what sort of changes a delta contains.

    :param delta: A TreeDelta to inspect
    :return: List with classes found (see classify_filename)
    