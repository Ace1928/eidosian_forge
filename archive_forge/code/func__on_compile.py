import copy
import re
import types
from .ucre import build_re
def _on_compile(self):
    """Override to modify basic RegExp-s."""
    pass