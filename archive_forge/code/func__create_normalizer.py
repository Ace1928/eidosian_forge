import copy
import re
import types
from .ucre import build_re
def _create_normalizer(self):

    def func(match):
        self.normalize(match)
    return func