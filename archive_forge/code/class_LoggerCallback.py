import abc
import json
import logging
import os
import pyarrow
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Type
import yaml
from ray.air._internal.json import SafeFallbackEncoder
from ray.tune.callback import Callback
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@PublicAPI
class LoggerCallback(Callback):
    """Base class for experiment-level logger callbacks

    This base class defines a general interface for logging events,
    like trial starts, restores, ends, checkpoint saves, and receiving
    trial results.

    Callbacks implementing this interface should make sure that logging
    utilities are cleaned up properly on trial termination, i.e. when
    ``log_trial_end`` is received. This includes e.g. closing files.
    """

    def log_trial_start(self, trial: 'Trial'):
        """Handle logging when a trial starts.

        Args:
            trial: Trial object.
        """
        pass

    def log_trial_restore(self, trial: 'Trial'):
        """Handle logging when a trial restores.

        Args:
            trial: Trial object.
        """
        pass

    def log_trial_save(self, trial: 'Trial'):
        """Handle logging when a trial saves a checkpoint.

        Args:
            trial: Trial object.
        """
        pass

    def log_trial_result(self, iteration: int, trial: 'Trial', result: Dict):
        """Handle logging when a trial reports a result.

        Args:
            trial: Trial object.
            result: Result dictionary.
        """
        pass

    def log_trial_end(self, trial: 'Trial', failed: bool=False):
        """Handle logging when a trial ends.

        Args:
            trial: Trial object.
            failed: True if the Trial finished gracefully, False if
                it failed (e.g. when it raised an exception).
        """
        pass

    def on_trial_result(self, iteration: int, trials: List['Trial'], trial: 'Trial', result: Dict, **info):
        self.log_trial_result(iteration, trial, result)

    def on_trial_start(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        self.log_trial_start(trial)

    def on_trial_restore(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        self.log_trial_restore(trial)

    def on_trial_save(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        self.log_trial_save(trial)

    def on_trial_complete(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        self.log_trial_end(trial, failed=False)

    def on_trial_error(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        self.log_trial_end(trial, failed=True)

    def _restore_from_remote(self, file_name: str, trial: 'Trial') -> None:
        if not trial.checkpoint:
            return
        local_file = os.path.join(trial.local_path, file_name)
        remote_file = os.path.join(trial.storage.trial_fs_path, file_name)
        try:
            pyarrow.fs.copy_files(remote_file, local_file, source_filesystem=trial.storage.storage_filesystem)
            logger.debug(f'Copied {remote_file} to {local_file}')
        except FileNotFoundError:
            logger.warning(f'Remote file not found: {remote_file}')
        except Exception:
            logger.exception(f'Error downloading {remote_file}')