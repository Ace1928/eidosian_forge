import builtins
import copy
import json
import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, Optional
import colorama
import ray
from ray._private.ray_constants import env_bool
from ray.util.debug import log_once
class _Bar:
    """Manages a single virtual progress bar on the driver.

    The actual position of individual bars is calculated as (pos_offset + position),
    where `pos_offset` is the position offset determined by the BarManager.
    """

    def __init__(self, state: ProgressBarState, pos_offset: int):
        """Initialize a bar.

        Args:
            state: The initial progress bar state.
            pos_offset: The position offset determined by the BarManager.
        """
        self.state = state
        self.pos_offset = pos_offset
        self.bar = real_tqdm.tqdm(desc=state['desc'] + ' ' + str(state['pos']), total=state['total'], position=pos_offset + state['pos'], leave=False)
        if state['x']:
            self.bar.update(state['x'])

    def update(self, state: ProgressBarState) -> None:
        """Apply the updated worker progress bar state."""
        if state['desc'] != self.state['desc']:
            self.bar.set_description(state['desc'])
        if state['total'] != self.state['total']:
            self.bar.total = state['total']
            self.bar.refresh()
        delta = state['x'] - self.state['x']
        if delta:
            self.bar.update(delta)
        self.state = state

    def close(self):
        """The progress bar has been closed."""
        self.bar.close()

    def update_offset(self, pos_offset: int) -> None:
        """Update the position offset assigned by the BarManager."""
        if pos_offset != self.pos_offset:
            self.pos_offset = pos_offset
            self.bar.clear()
            self.bar.pos = -(pos_offset + self.state['pos'])
            self.bar.refresh()