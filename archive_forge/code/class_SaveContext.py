import contextlib
import threading
class SaveContext(threading.local):
    """A context for building a graph of SavedModel."""

    def __init__(self):
        super(SaveContext, self).__init__()
        self._in_save_context = False
        self._options = None

    def options(self):
        if not self.in_save_context():
            raise ValueError('Not in a SaveContext.')
        return self._options

    def enter_save_context(self, options):
        self._in_save_context = True
        self._options = options

    def exit_save_context(self):
        self._in_save_context = False
        self._options = None

    def in_save_context(self):
        return self._in_save_context