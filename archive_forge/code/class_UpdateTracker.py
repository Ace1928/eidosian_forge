import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
class UpdateTracker:
    """Tracks updates to an optimization model from a ModelStorage.

    Do not instantiate directly, instead create through
    ModelStorage.add_update_tracker().

    Querying an UpdateTracker after calling Model.remove_update_tracker will
    result in a model_storage.UsedUpdateTrackerAfterRemovalError.

    Example:
      mod = Model()
      x = mod.add_variable(0.0, 1.0, True, 'x')
      y = mod.add_variable(0.0, 1.0, True, 'y')
      tracker = mod.add_update_tracker()
      mod.set_variable_ub(x, 3.0)
      tracker.export_update()
        => "variable_updates: {upper_bounds: {ids: [0], values[3.0] }"
      mod.set_variable_ub(y, 2.0)
      tracker.export_update()
        => "variable_updates: {upper_bounds: {ids: [0, 1], values[3.0, 2.0] }"
      tracker.advance_checkpoint()
      tracker.export_update()
        => None
      mod.set_variable_ub(y, 4.0)
      tracker.export_update()
        => "variable_updates: {upper_bounds: {ids: [1], values[4.0] }"
      tracker.advance_checkpoint()
      mod.remove_update_tracker(tracker)
    """

    def __init__(self, storage_update_tracker: model_storage.StorageUpdateTracker):
        """Do not invoke directly, use Model.add_update_tracker() instead."""
        self.storage_update_tracker = storage_update_tracker

    def export_update(self) -> Optional[model_update_pb2.ModelUpdateProto]:
        """Returns changes to the model since last call to checkpoint/creation."""
        return self.storage_update_tracker.export_update()

    def advance_checkpoint(self) -> None:
        """Track changes to the model only after this function call."""
        return self.storage_update_tracker.advance_checkpoint()