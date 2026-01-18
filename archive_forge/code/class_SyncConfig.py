import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
@PublicAPI(stability='stable')
@dataclass
class SyncConfig:
    """Configuration object for Train/Tune file syncing to `RunConfig(storage_path)`.

    In Ray Train/Tune, here is where syncing (mainly uploading) happens:

    The experiment driver (on the head node) syncs the experiment directory to storage
    (which includes experiment state such as searcher state, the list of trials
    and their statuses, and trial metadata).

    It's also possible to sync artifacts from the trial directory to storage
    by setting `sync_artifacts=True`.
    For a Ray Tune run with many trials, each trial will upload its trial directory
    to storage, which includes arbitrary files that you dumped during the run.
    For a Ray Train run doing distributed training, each remote worker will similarly
    upload its trial directory to storage.

    See :ref:`persistent-storage-guide` for more details and examples.

    Args:
        sync_period: Minimum time in seconds to wait between two sync operations.
            A smaller ``sync_period`` will have the data in storage updated more often
            but introduces more syncing overhead. Defaults to 5 minutes.
        sync_timeout: Maximum time in seconds to wait for a sync process
            to finish running. A sync operation will run for at most this long
            before raising a `TimeoutError`. Defaults to 30 minutes.
        sync_artifacts: [Beta] Whether or not to sync artifacts that are saved to the
            trial directory (accessed via `train.get_context().get_trial_dir()`)
            to the persistent storage configured via `train.RunConfig(storage_path)`.
            The trial or remote worker will try to launch an artifact syncing
            operation every time `train.report` happens, subject to `sync_period`
            and `sync_artifacts_on_checkpoint`.
            Defaults to False -- no artifacts are persisted by default.
        sync_artifacts_on_checkpoint: If True, trial/worker artifacts are
            forcefully synced on every reported checkpoint.
            This only has an effect if `sync_artifacts` is True.
            Defaults to True.
    """
    sync_period: int = DEFAULT_SYNC_PERIOD
    sync_timeout: int = DEFAULT_SYNC_TIMEOUT
    sync_artifacts: bool = False
    sync_artifacts_on_checkpoint: bool = True
    upload_dir: Optional[str] = _DEPRECATED_VALUE
    syncer: Optional[Union[str, 'Syncer']] = _DEPRECATED_VALUE
    sync_on_checkpoint: bool = _DEPRECATED_VALUE

    def _deprecation_warning(self, attr_name: str, extra_msg: str):
        if getattr(self, attr_name) != _DEPRECATED_VALUE:
            if log_once(f'sync_config_param_deprecation_{attr_name}'):
                warnings.warn(f'`SyncConfig({attr_name})` is a deprecated configuration and will be ignored. Please remove it from your `SyncConfig`, as this will raise an error in a future version of Ray.{extra_msg}')

    def __post_init__(self):
        for attr_name, extra_msg in [('upload_dir', '\nPlease specify `train.RunConfig(storage_path)` instead.'), ('syncer', '\nPlease implement custom syncing logic with a custom `pyarrow.fs.FileSystem` instead, and pass it into `train.RunConfig(storage_filesystem)`. See here: https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html#custom-storage'), ('sync_on_checkpoint', '')]:
            self._deprecation_warning(attr_name, extra_msg)

    def _repr_html_(self) -> str:
        """Generate an HTML representation of the SyncConfig."""
        return Template('scrollableTable.html.j2').render(table=tabulate({'Setting': ['Sync period', 'Sync timeout'], 'Value': [self.sync_period, self.sync_timeout]}, tablefmt='html', showindex=False, headers='keys'), max_height='none')