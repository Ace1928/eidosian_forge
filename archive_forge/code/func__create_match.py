import copy
import re
import types
from .ucre import build_re
def _create_match(self, shift):
    match = Match(self, shift)
    self._compiled[match.schema]['normalize'](match)
    return match