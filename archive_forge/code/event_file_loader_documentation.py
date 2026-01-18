import contextlib
from tensorboard import data_compat
from tensorboard import dataclass_compat
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.util import platform_util
from tensorboard.util import tb_logging
Loads all new events and their wall time values from disk.

        Calling Load multiple times in a row will not 'drop' events as long as the
        return value is not iterated over.

        Yields:
          Pairs of (UNIX timestamp float, Event proto) for all events in the file
          that have not been yielded yet.
        