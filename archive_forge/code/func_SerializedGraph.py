import os
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
def SerializedGraph(self, run):
    """Retrieve the serialized graph associated with the provided run.

        Args:
          run: A string name of a run to load the graph for.

        Raises:
          KeyError: If the run is not found.
          ValueError: If the run does not have an associated graph.

        Returns:
          The serialized form of the `GraphDef` protobuf data structure.
        """
    accumulator = self.GetAccumulator(run)
    return accumulator.SerializedGraph()