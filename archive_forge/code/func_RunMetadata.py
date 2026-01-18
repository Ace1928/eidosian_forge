import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def RunMetadata(self, run, tag):
    """Get the session.run() metadata associated with a TensorFlow run and
        tag.

        Args:
          run: A string name of a TensorFlow run.
          tag: A string name of the tag associated with a particular session.run().

        Raises:
          KeyError: If the run is not found, or the tag is not available for the
            given run.

        Returns:
          The metadata in the form of `RunMetadata` protobuf data structure.
        """
    accumulator = self.GetAccumulator(run)
    return accumulator.RunMetadata(tag)