import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
class DVCLiveTracker(GeneralTracker):
    """
    A `Tracker` class that supports `dvclive`. Should be initialized at the start of your script.

    Args:
        run_name (`str`, *optional*):
            Ignored for dvclive. See `kwargs` instead.
        kwargs:
            Additional key word arguments passed along to [`dvclive.Live()`](https://dvc.org/doc/dvclive/live).

    Example:

    ```py
    from accelerate import Accelerator

    accelerator = Accelerator(log_with="dvclive")
    accelerator.init_trackers(project_name="my_project", init_kwargs={"dvclive": {"dir": "my_directory"}})
    ```
    """
    name = 'dvclive'
    requires_logging_directory = False

    @on_main_process
    def __init__(self, run_name: Optional[str]=None, live: Optional[Any]=None, **kwargs):
        from dvclive import Live
        super().__init__()
        self.live = live if live is not None else Live(**kwargs)

    @property
    def tracker(self):
        return self.live

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment. Stores the
        hyperparameters in a yaml file for future use.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float`, `int`, or a List or Dict of those types):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, or `int`.
        """
        self.live.log_params(values)

    @on_main_process
    def log(self, values: dict, step: Optional[int]=None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, or `int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, or `int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to `dvclive.Live.log_metric()`.
        """
        from dvclive.plots import Metric
        if step is not None:
            self.live.step = step
        for k, v in values.items():
            if Metric.could_log(v):
                self.live.log_metric(k, v, **kwargs)
            else:
                logger.warning_once(f'''Accelerator attempted to log a value of "{v}" of type {type(v)} for key "{k}" as a scalar. This invocation of DVCLive's Live.log_metric() is incorrect so we dropped this attribute.''')
        self.live.next_step()

    @on_main_process
    def finish(self):
        """
        Closes `dvclive.Live()`.
        """
        self.live.end()