import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
class BaseDigest:
    """Base class for digest.

  Properties:
    wall_time: A timestamp for the digest as a `float` (unit: s).
    locator: A datum that allows tracng the digest to its original
      location. It can be either of the two:
       1. Bytes offset from the beginning of the file as a single integer,
          for the case of all digests of the same kind coming from the same
          file.
       2. A tuple of a file index and a byte offset. This applies to case
          in which the same type of debugger data may come from multple files,
          e.g., graph execution traces.
  """

    def __init__(self, wall_time, locator):
        self._wall_time = wall_time
        self._locator = locator

    @property
    def wall_time(self):
        return self._wall_time

    @property
    def locator(self):
        return self._locator

    def to_json(self):
        return {'wall_time': self.wall_time}