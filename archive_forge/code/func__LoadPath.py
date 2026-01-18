from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import tb_logging
def _LoadPath(self, path):
    """Generator for values from a single path's loader.

        Args:
          path: the path to load from

        Yields:
          All values from this path's loader that have not been yielded yet.
        """
    max_timestamp = self._max_timestamps.get(path, None)
    if max_timestamp is _INACTIVE or self._MarkIfInactive(path, max_timestamp):
        logger.debug('Skipping inactive path %s', path)
        return
    loader = self._loaders.get(path, None)
    if loader is None:
        try:
            loader = self._loader_factory(path)
        except tf.errors.NotFoundError:
            logger.debug('Skipping nonexistent path %s', path)
            return
        self._loaders[path] = loader
    logger.info('Loading data from path %s', path)
    for timestamp, value in loader.Load():
        if max_timestamp is None or timestamp > max_timestamp:
            max_timestamp = timestamp
        yield value
    if not self._MarkIfInactive(path, max_timestamp):
        self._max_timestamps[path] = max_timestamp