import collections
from absl import logging
def _set_mutable(self, mutable):
    """Change the mutability property to `mutable`."""
    object.__setattr__(self, '_mutable', mutable)