import functools
from typing import Optional
from .base import VariableTracker
def _create_realize_and_forward(name):

    @functools.wraps(getattr(VariableTracker, name))
    def realize_and_forward(self, *args, **kwargs):
        return getattr(self.realize(), name)(*args, **kwargs)
    return realize_and_forward