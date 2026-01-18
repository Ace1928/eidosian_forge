import sys
import fixtures
from functools import wraps
def _teardown_decorator(self):
    if hasattr(self.conf.set_override, 'wrapped'):
        self.conf.set_override = self.conf.set_override.wrapped
    if hasattr(self.conf.clear_override, 'wrapped'):
        self.conf.clear_override = self.conf.clear_override.wrapped