import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def SummaryMetadata(self, run, tag):
    """Return the summary metadata for the given tag on the given run.

        Args:
          run: A string name of the run for which summary metadata is to be
            retrieved.
          tag: A string name of the tag whose summary metadata is to be
            retrieved.

        Raises:
          KeyError: If the run is not found, or the tag is not available for
            the given run.

        Returns:
          A `SummaryMetadata` protobuf.
        """
    accumulator = self.GetAccumulator(run)
    return accumulator.SummaryMetadata(tag)