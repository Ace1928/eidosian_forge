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