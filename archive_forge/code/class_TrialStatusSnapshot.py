from collections import defaultdict
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
class TrialStatusSnapshot:
    """A sequence of statuses of trials as they progress.

    If all trials keep previous status, no snapshot is taken.
    """

    def __init__(self):
        self._snapshot = []

    def append(self, new_snapshot: Dict[str, str]):
        """May append a new snapshot to the sequence."""
        if not new_snapshot:
            return
        if not self._snapshot or new_snapshot != self._snapshot[-1]:
            self._snapshot.append(new_snapshot)

    def max_running_trials(self) -> int:
        """Outputs the max number of running trials at a given time.

        Usually used to assert certain number given resource restrictions.
        """
        result = 0
        for snapshot in self._snapshot:
            count = 0
            for trial_id in snapshot:
                if snapshot[trial_id] == Trial.RUNNING:
                    count += 1
            result = max(result, count)
        return result

    def all_trials_are_terminated(self) -> bool:
        """True if all trials are terminated."""
        if not self._snapshot:
            return False
        last_snapshot = self._snapshot[-1]
        return all((last_snapshot[trial_id] == Trial.TERMINATED for trial_id in last_snapshot))