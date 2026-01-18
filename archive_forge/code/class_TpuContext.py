import contextlib
import threading
class TpuContext(threading.local):
    """A context object holding state about the TPU computation being built."""

    def __init__(self):
        """Creates a new TpuContext."""
        self._number_of_shards = None

    @property
    def number_of_shards(self):
        return self._number_of_shards

    def set_number_of_shards(self, number_of_shards):
        self._number_of_shards = number_of_shards