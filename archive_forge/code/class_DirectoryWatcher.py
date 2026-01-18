import bisect
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
class DirectoryWatcher:
    """A DirectoryWatcher wraps a loader to load from a sequence of paths.

    A loader reads a path and produces some kind of values as an iterator. A
    DirectoryWatcher takes a directory, a factory for loaders, and optionally a
    path filter and watches all the paths inside that directory.

    This class is only valid under the assumption that only one path will be
    written to by the data source at a time and that once the source stops writing
    to a path, it will start writing to a new path that's lexicographically
    greater and never come back. It uses some heuristics to check whether this is
    true based on tracking changes to the files' sizes, but the check can have
    false negatives. However, it should have no false positives.
    """

    def __init__(self, directory, loader_factory, path_filter=lambda x: True):
        """Constructs a new DirectoryWatcher.

        Args:
          directory: The directory to load files from.
          loader_factory: A factory for creating loaders. The factory should take a
            path and return an object that has a Load method returning an
            iterator that will yield all events that have not been yielded yet.
          path_filter: If specified, only paths matching this filter are loaded.

        Raises:
          ValueError: If path_provider or loader_factory are None.
        """
        if directory is None:
            raise ValueError('A directory is required')
        if loader_factory is None:
            raise ValueError('A loader factory is required')
        self._directory = directory
        self._path = None
        self._loader_factory = loader_factory
        self._loader = None
        self._path_filter = path_filter
        self._ooo_writes_detected = False
        self._finalized_sizes = {}

    def Load(self):
        """Loads new values.

        The watcher will load from one path at a time; as soon as that path stops
        yielding events, it will move on to the next path. We assume that old paths
        are never modified after a newer path has been written. As a result, Load()
        can be called multiple times in a row without losing events that have not
        been yielded yet. In other words, we guarantee that every event will be
        yielded exactly once.

        Yields:
          All values that have not been yielded yet.

        Raises:
          DirectoryDeletedError: If the directory has been permanently deleted
            (as opposed to being temporarily unavailable).
        """
        try:
            for event in self._LoadInternal():
                yield event
        except tf.errors.OpError:
            if not tf.io.gfile.exists(self._directory):
                raise DirectoryDeletedError('Directory %s has been permanently deleted' % self._directory)

    def _LoadInternal(self):
        """Internal implementation of Load().

        The only difference between this and Load() is that the latter will throw
        DirectoryDeletedError on I/O errors if it thinks that the directory has been
        permanently deleted.

        Yields:
          All values that have not been yielded yet.
        """
        if not self._loader:
            self._InitializeLoader()
        if not self._loader:
            return
        while True:
            for event in self._loader.Load():
                yield event
            next_path = self._GetNextPath()
            if not next_path:
                logger.info('No path found after %s', self._path)
                return
            for event in self._loader.Load():
                yield event
            logger.info('Directory watcher advancing from %s to %s', self._path, next_path)
            self._SetPath(next_path)
    _OOO_WRITE_CHECK_COUNT = 20

    def OutOfOrderWritesDetected(self):
        """Returns whether any out-of-order writes have been detected.

        Out-of-order writes are only checked as part of the Load() iterator. Once an
        out-of-order write is detected, this function will always return true.

        Note that out-of-order write detection is not performed on GCS paths, so
        this function will always return false.

        Returns:
          Whether any out-of-order write has ever been detected by this watcher.
        """
        return self._ooo_writes_detected

    def _InitializeLoader(self):
        path = self._GetNextPath()
        if path:
            self._SetPath(path)

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

    def _GetNextPath(self):
        """Gets the next path to load from.

        This function also does the checking for out-of-order writes as it iterates
        through the paths.

        Returns:
          The next path to load events from, or None if there are no more paths.
        """
        paths = sorted((path for path in io_wrapper.ListDirectoryAbsolute(self._directory) if self._path_filter(path)))
        if not paths:
            return None
        if self._path is None:
            return paths[0]
        if not io_util.IsCloudPath(paths[0]) and (not self._ooo_writes_detected):
            current_path_index = bisect.bisect_left(paths, self._path)
            ooo_check_start = max(0, current_path_index - self._OOO_WRITE_CHECK_COUNT)
            for path in paths[ooo_check_start:current_path_index]:
                if self._HasOOOWrite(path):
                    self._ooo_writes_detected = True
                    break
        next_paths = list((path for path in paths if self._path is None or path > self._path))
        if next_paths:
            return min(next_paths)
        else:
            return None

    def _HasOOOWrite(self, path):
        """Returns whether the path has had an out-of-order write."""
        size = tf.io.gfile.stat(path).length
        old_size = self._finalized_sizes.get(path, None)
        if size != old_size:
            if old_size is None:
                logger.error("File %s created after file %s even though it's lexicographically earlier", path, self._path)
            else:
                logger.error('File %s updated even though the current file is %s', path, self._path)
            return True
        else:
            return False