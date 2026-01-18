from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import click
import logging
import os
import time
import warnings
from ray.train._internal.storage import (
from ray.tune.experiment import Trial
from ray.tune.impl.out_of_band_serialize_dataset import out_of_band_serialize_dataset
def _resume_auto(self) -> bool:
    experiment_local_path = self._storage.experiment_local_path
    experiment_fs_path = self._storage.experiment_fs_path
    syncer = self._storage.syncer
    if experiment_fs_path and syncer:
        logger.info(f'Trying to find and download experiment checkpoint at {experiment_fs_path}')
        try:
            self.sync_down_experiment_state()
        except Exception:
            logger.exception("Got error when trying to sync down.\nPlease check this error message for potential access problems - if a directory was not found, that is expected at this stage when you're starting a new experiment.")
            logger.info('No remote checkpoint was found or an error occurred when trying to download the experiment checkpoint. Please check the previous warning message for more details. Starting a new run...')
            return False
        if not _experiment_checkpoint_exists(experiment_local_path):
            logger.warning('A remote checkpoint was fetched, but no checkpoint data was found. This can happen when e.g. the cloud bucket exists but does not contain any data. Starting a new run...')
            return False
        logger.info('A remote experiment checkpoint was found and will be used to restore the previous experiment state.')
        return True
    elif not _experiment_checkpoint_exists(experiment_local_path):
        logger.info('No local checkpoint was found. Starting a new run...')
        return False
    logger.info('A local experiment checkpoint was found and will be used to restore the previous experiment state.')
    return True