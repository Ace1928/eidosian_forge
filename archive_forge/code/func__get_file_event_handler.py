import abc
import fnmatch
import glob
import logging
import os
import queue
import time
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, MutableSet, Optional
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib.paths import LogicalPath
def _get_file_event_handler(self, file_path: PathStr, save_name: LogicalPath) -> FileEventHandler:
    """Get or create an event handler for a particular file.

        file_path: the file's actual path
        save_name: its path relative to the run directory (aka the watch directory)
        """
    if save_name.startswith('media/'):
        return PolicyNow(file_path, save_name, self._file_pusher, self._settings)
    if save_name not in self._file_event_handlers:
        if 'tfevents' in save_name or 'graph.pbtxt' in save_name:
            self._file_event_handlers[save_name] = PolicyLive(file_path, save_name, self._file_pusher, self._settings)
        elif save_name in self._savename_file_policies:
            policy_name = self._savename_file_policies[save_name]
            make_handler = PolicyLive if policy_name == 'live' else PolicyNow if policy_name == 'now' else PolicyEnd
            self._file_event_handlers[save_name] = make_handler(file_path, save_name, self._file_pusher, self._settings)
        else:
            make_handler = PolicyEnd
            for policy, globs in self._user_file_policies.items():
                if policy == 'end':
                    continue
                for g in list(globs):
                    paths = glob.glob(os.path.join(self._dir, g))
                    if any((save_name in p for p in paths)):
                        if policy == 'live':
                            make_handler = PolicyLive
                        elif policy == 'now':
                            make_handler = PolicyNow
            self._file_event_handlers[save_name] = make_handler(file_path, save_name, self._file_pusher, self._settings)
    return self._file_event_handlers[save_name]