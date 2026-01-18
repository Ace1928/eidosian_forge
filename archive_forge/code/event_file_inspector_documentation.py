import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
Main function for inspector that prints out a digest of event files.

    Args:
      logdir: A log directory that contains event files.
      event_file: Or, a particular event file path.
      tag: An optional tag name to query for.

    Raises:
      ValueError: If neither logdir and event_file are given, or both are given.
    