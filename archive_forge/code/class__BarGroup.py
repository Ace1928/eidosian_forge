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
class _BarGroup:
    """Manages a group of virtual progress bar produced by a single worker.

    All the progress bars in the group have the same `pos_offset` determined by the
    BarManager for the process.
    """

    def __init__(self, ip, pid, pos_offset):
        self.ip = ip
        self.pid = pid
        self.pos_offset = pos_offset
        self.bars_by_uuid: Dict[str, _Bar] = {}

    def has_bar(self, bar_uuid) -> bool:
        """Return whether this bar exists."""
        return bar_uuid in self.bars_by_uuid

    def allocate_bar(self, state: ProgressBarState) -> None:
        """Add a new bar to this group."""
        self.bars_by_uuid[state['uuid']] = _Bar(state, self.pos_offset)

    def update_bar(self, state: ProgressBarState) -> None:
        """Update the state of a managed bar in this group."""
        bar = self.bars_by_uuid[state['uuid']]
        bar.update(state)

    def close_bar(self, state: ProgressBarState) -> None:
        """Remove a bar from this group."""
        bar = self.bars_by_uuid[state['uuid']]
        bar.close()
        del self.bars_by_uuid[state['uuid']]

    def slots_required(self):
        """Return the number of pos slots we need to accomodate bars in this group."""
        if not self.bars_by_uuid:
            return 0
        return 1 + max((bar.state['pos'] for bar in self.bars_by_uuid.values()))

    def update_offset(self, offset: int) -> None:
        """Update the position offset assigned by the BarManager."""
        if offset != self.pos_offset:
            self.pos_offset = offset
            for bar in self.bars_by_uuid.values():
                bar.update_offset(offset)

    def hide_bars(self) -> None:
        """Temporarily hide visible bars to avoid conflict with other log messages."""
        for bar in self.bars_by_uuid.values():
            bar.bar.clear()

    def unhide_bars(self) -> None:
        """Opposite of hide_bars()."""
        for bar in self.bars_by_uuid.values():
            bar.bar.refresh()