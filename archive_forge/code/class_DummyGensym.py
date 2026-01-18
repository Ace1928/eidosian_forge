import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
class DummyGensym:
    """A dumb gensym that suffixes a stem by sequential numbers from 1000."""

    def __init__(self):
        self._idx = 0

    def new_name(self, stem='tmp'):
        self._idx += 1
        return stem + '_' + str(1000 + self._idx)