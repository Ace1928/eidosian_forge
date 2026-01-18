import bisect
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def _SetPath(self, path):
    """Sets the current path to watch for new events.

        This also records the size of the old path, if any. If the size can't be
        found, an error is logged.

        Args:
          path: The full path of the file to watch.
        """
    old_path = self._path
    if old_path and (not io_util.IsCloudPath(old_path)):
        try:
            size = tf.io.gfile.stat(old_path).length
            logger.debug('Setting latest size of %s to %d', old_path, size)
            self._finalized_sizes[old_path] = size
        except tf.errors.OpError as e:
            logger.error('Unable to get size of %s: %s', old_path, e)
    self._path = path
    self._loader = self._loader_factory(path)