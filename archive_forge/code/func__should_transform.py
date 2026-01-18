import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _should_transform(self, parent, field, child):
    for pat, result in self._overrides:
        if self._match(pat, parent, field, child):
            return result(parent, field, child)
    return False