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