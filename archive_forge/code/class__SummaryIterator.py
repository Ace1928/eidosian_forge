from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util.tf_export import tf_export
class _SummaryIterator(object):
    """Yields `Event` protocol buffers from a given path."""

    def __init__(self, path):
        self._tf_record_iterator = tf_record.tf_record_iterator(path)

    def __iter__(self):
        return self

    def __next__(self):
        r = next(self._tf_record_iterator)
        return event_pb2.Event.FromString(r)
    next = __next__