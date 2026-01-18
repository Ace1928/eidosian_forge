import ast
import sys
class _CatchDisplay:
    """Class to temporarily catch sys.displayhook"""

    def __init__(self):
        self.output = None

    def __enter__(self):
        self.old_hook = sys.displayhook
        sys.displayhook = self
        return self

    def __exit__(self, type, value, traceback):
        sys.displayhook = self.old_hook
        return False

    def __call__(self, output):
        self.output = output