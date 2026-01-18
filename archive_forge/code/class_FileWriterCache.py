import threading
from tensorflow.python.framework import ops
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['summary.FileWriterCache'])
class FileWriterCache(object):
    """Cache for file writers.

  This class caches file writers, one per directory.
  """
    _cache = {}
    _lock = threading.RLock()

    @staticmethod
    def clear():
        """Clear cached summary writers. Currently only used for unit tests."""
        with FileWriterCache._lock:
            for item in FileWriterCache._cache.values():
                item.close()
            FileWriterCache._cache = {}

    @staticmethod
    def get(logdir):
        """Returns the FileWriter for the specified directory.

    Args:
      logdir: str, name of the directory.

    Returns:
      A `FileWriter`.
    """
        with FileWriterCache._lock:
            if logdir not in FileWriterCache._cache:
                FileWriterCache._cache[logdir] = FileWriter(logdir, graph=ops.get_default_graph())
            return FileWriterCache._cache[logdir]