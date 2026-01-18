from tensorflow.python import pywrap_tfe
class CancellationManagerContext:
    """A Python context for wrapping a cancellable ConcreteFunction."""

    def __init__(self, cancellation_manager):
        self._cancellation_manager = cancellation_manager

    def __enter__(self):
        global _active_context
        _active_context = self._cancellation_manager

    def __exit__(self, exc_type, exc_value, exc_tb):
        global _active_context
        _active_context = None