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
class _BarManager:
    """Central tqdm manager run on the driver.

    This class holds a collection of BarGroups and updates their `pos_offset` as
    needed to ensure individual progress bars do not collide in position, kind of
    like a virtual memory manager.
    """

    def __init__(self):
        import ray._private.services as services
        self.ip = services.get_node_ip_address()
        self.pid = os.getpid()
        self.bar_groups = {}
        self.in_hidden_state = False
        self.num_hides = 0
        self.lock = threading.RLock()
        self.should_colorize = not ray.widgets.util.in_notebook()

    def process_state_update(self, state: ProgressBarState) -> None:
        """Apply the remote progress bar state update.

        This creates a new bar locally if it doesn't already exist. When a bar is
        created or destroyed, we also recalculate and update the `pos_offset` of each
        BarGroup on the screen.
        """
        with self.lock:
            self._process_state_update_locked(state)

    def _process_state_update_locked(self, state: ProgressBarState) -> None:
        if not real_tqdm:
            if log_once('no_tqdm'):
                logger.warning('tqdm is not installed. Progress bars will be disabled.')
            return
        if state['ip'] == self.ip:
            if state['pid'] == self.pid:
                prefix = ''
            else:
                prefix = '(pid={}) '.format(state.get('pid'))
                if self.should_colorize:
                    prefix = '{}{}{}{}'.format(colorama.Style.DIM, colorama.Fore.CYAN, prefix, colorama.Style.RESET_ALL)
        else:
            prefix = '(pid={}, ip={}) '.format(state.get('pid'), state.get('ip'))
            if self.should_colorize:
                prefix = '{}{}{}{}'.format(colorama.Style.DIM, colorama.Fore.CYAN, prefix, colorama.Style.RESET_ALL)
        state['desc'] = prefix + state['desc']
        process = self._get_or_allocate_bar_group(state)
        if process.has_bar(state['uuid']):
            if state['closed']:
                process.close_bar(state)
                self._update_offsets()
            else:
                process.update_bar(state)
        else:
            process.allocate_bar(state)
            self._update_offsets()

    def hide_bars(self) -> None:
        """Temporarily hide visible bars to avoid conflict with other log messages."""
        with self.lock:
            if not self.in_hidden_state:
                self.in_hidden_state = True
                self.num_hides += 1
                for group in self.bar_groups.values():
                    group.hide_bars()

    def unhide_bars(self) -> None:
        """Opposite of hide_bars()."""
        with self.lock:
            if self.in_hidden_state:
                self.in_hidden_state = False
                for group in self.bar_groups.values():
                    group.unhide_bars()

    def _get_or_allocate_bar_group(self, state: ProgressBarState):
        ptuple = (state['ip'], state['pid'])
        if ptuple not in self.bar_groups:
            offset = sum((p.slots_required() for p in self.bar_groups.values()))
            self.bar_groups[ptuple] = _BarGroup(state['ip'], state['pid'], offset)
        return self.bar_groups[ptuple]

    def _update_offsets(self):
        offset = 0
        for proc in self.bar_groups.values():
            proc.update_offset(offset)
            offset += proc.slots_required()