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
Checks whether to resume experiment.

        If experiment should be resumed, this method may sync down experiment state
        from the cloud and then return a ResumeConfig mapping to the resume type.

        Args:
            resume_type: One of ["REMOTE", "LOCAL", "PROMPT", "AUTO"]. Can
                be suffixed with one or more of ["+ERRORED", "+ERRORED_ONLY",
                "+RESTART_ERRORED", "+RESTART_ERRORED_ONLY"]

        Returns:
            _ResumeConfig if resume is successful. None otherwise.
        