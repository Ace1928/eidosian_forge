import re
from .. import osutils
from ..iterablefile import IterableFile
class RioReader:
    """Read stanzas from a file as a sequence

    to_file can be anything that can be enumerated as a sequence of
    lines (with newlines.)
    """

    def __init__(self, from_file):
        self._from_file = from_file

    def __iter__(self):
        while True:
            s = read_stanza(self._from_file)
            if s is None:
                break
            else:
                yield s