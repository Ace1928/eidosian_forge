import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def _vertScroll(self, n):
    self._viewport.yOffset += n
    self._viewport.yOffset = max(0, self._viewport.yOffset)
    return self._viewport.yOffset / 25.0