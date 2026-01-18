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
def _update_auto_checkpoint_time(self, time_taken: float):
    if self._auto_checkpoint_enabled:
        self._checkpoint_period = max(10.0, time_taken * 19)
        logger.debug(f'Global experiment checkpointing took {time_taken:.2f} seconds. Adjusting checkpoint period to {self._checkpoint_period:.2f} seconds.')