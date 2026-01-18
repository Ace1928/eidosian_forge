import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
class GeneralTracker:
    """
    A base Tracker class to be used for all logging integration implementations.

    Each function should take in `**kwargs` that will automatically be passed in from a base dictionary provided to
    [`Accelerator`].

    Should implement `name`, `requires_logging_directory`, and `tracker` properties such that:

    `name` (`str`): String representation of the tracker class name, such as "TensorBoard" `requires_logging_directory`
    (`bool`): Whether the logger requires a directory to store their logs. `tracker` (`object`): Should return internal
    tracking mechanism used by a tracker class (such as the `run` for wandb)

    Implementations can also include a `main_process_only` (`bool`) attribute to toggle if relevent logging, init, and
    other functions should occur on the main process or across all processes (by default will use `True`)
    """
    main_process_only = True

    def __init__(self, _blank=False):
        if not _blank:
            err = ''
            if not hasattr(self, 'name'):
                err += '`name`'
            if not hasattr(self, 'requires_logging_directory'):
                if len(err) > 0:
                    err += ', '
                err += '`requires_logging_directory`'
            if 'tracker' not in dir(self):
                if len(err) > 0:
                    err += ', '
                err += '`tracker`'
            if len(err) > 0:
                raise NotImplementedError(f'The implementation for this tracker class is missing the following required attributes. Please define them in the class definition: {err}')

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Implementations should use the experiment configuration
        functionality of a tracking API.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        pass

    def log(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `values` to the current run. Base `log` implementations of a tracking API should go in here, along with
        special behavior for the `step parameter.

        Args:
            values (Dictionary `str` to `str`, `float`, or `int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, or `int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        pass

    def finish(self):
        """
        Should run any finalizing functions within the tracking API. If the API should not have one, just don't
        overwrite that method.
        """
        pass