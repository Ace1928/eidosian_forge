import contextlib
from tensorboard import data_compat
from tensorboard import dataclass_compat
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.util import platform_util
from tensorboard.util import tb_logging
class LegacyEventFileLoader(RawEventFileLoader):
    """An iterator that yields parsed Event protos."""

    def Load(self):
        """Loads all new events from disk.

        Calling Load multiple times in a row will not 'drop' events as long as the
        return value is not iterated over.

        Yields:
          All events in the file that have not been yielded yet.
        """
        for record in super().Load():
            yield event_pb2.Event.FromString(record)