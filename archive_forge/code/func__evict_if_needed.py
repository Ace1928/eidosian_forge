import logging
from typing import Set, Callable
def _evict_if_needed(self, logger: logging.Logger=default_logger):
    """Evict unused URIs (if they exist) until total size <= max size."""
    while self._unused_uris and self.get_total_size_bytes() > self.max_total_size_bytes:
        arbitrary_unused_uri = next(iter(self._unused_uris))
        self._unused_uris.remove(arbitrary_unused_uri)
        num_bytes_deleted = self._delete_fn(arbitrary_unused_uri, logger)
        self._total_size_bytes -= num_bytes_deleted
        logger.info(f'Deleted URI {arbitrary_unused_uri} with size {num_bytes_deleted}.')