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
def _resume_str_to_config(resume_str: str) -> Tuple[str, _ResumeConfig]:
    if resume_str is True:
        resume_str = 'LOCAL'
    elif resume_str == 'ERRORED_ONLY':
        warnings.warn("Passing `resume='ERRORED_ONLY'` to tune.run() is deprecated and will be removed in the future. Please pass e.g. `resume='LOCAL+RESTART_ERRORED_ONLY'` instead.")
        resume_str = 'LOCAL+RESTART_ERRORED_ONLY'
    resume_config = _ResumeConfig()
    resume_settings = resume_str.split('+')
    resume_str = resume_settings[0]
    for setting in resume_settings:
        if setting == 'ERRORED':
            resume_config.resume_errored = True
        elif setting == 'RESTART_ERRORED':
            resume_config.restart_errored = True
        elif setting == 'ERRORED_ONLY':
            resume_config.resume_unfinished = False
            resume_config.restart_errored = False
            resume_config.resume_errored = True
        elif setting == 'RESTART_ERRORED_ONLY':
            resume_config.resume_unfinished = False
            resume_config.restart_errored = True
            resume_config.resume_errored = False
    assert resume_str in VALID_RESUME_TYPES, 'resume={} is not one of {}'.format(resume_str, VALID_RESUME_TYPES)
    return (resume_str, resume_config)