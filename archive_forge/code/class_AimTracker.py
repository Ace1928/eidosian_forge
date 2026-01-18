import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
class AimTracker(GeneralTracker):
    """
    A `Tracker` class that supports `aim`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run.
        kwargs:
            Additional key word arguments passed along to the `Run.__init__` method.
    """
    name = 'aim'
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Optional[Union[str, os.PathLike]]='.', **kwargs):
        self.run_name = run_name
        from aim import Run
        self.writer = Run(repo=logging_dir, **kwargs)
        self.writer.name = self.run_name
        logger.debug(f'Initialized Aim project {self.run_name}')
        logger.debug('Make sure to log any initial configurations with `self.store_init_configuration` before training!')

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs.
        """
        self.writer['hparams'] = values

    @on_main_process
    def log(self, values: dict, step: Optional[int], **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (`dict`):
                Values to be logged as key-value pairs.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `Run.track` method.
        """
        for key, value in values.items():
            self.writer.track(value, name=key, step=step, **kwargs)

    @on_main_process
    def log_images(self, values: dict, step: Optional[int]=None, kwargs: Optional[Dict[str, dict]]=None):
        """
        Logs `images` to the current run.

        Args:
            values (`Dict[str, Union[np.ndarray, PIL.Image, Tuple[np.ndarray, str], Tuple[PIL.Image, str]]]`):
                Values to be logged as key-value pairs. The values need to have type `np.ndarray` or PIL.Image. If a
                tuple is provided, the first element should be the image and the second element should be the caption.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs (`Dict[str, dict]`):
                Additional key word arguments passed along to the `Run.Image` and `Run.track` method specified by the
                keys `aim_image` and `track`, respectively.
        """
        import aim
        aim_image_kw = {}
        track_kw = {}
        if kwargs is not None:
            aim_image_kw = kwargs.get('aim_image', {})
            track_kw = kwargs.get('track', {})
        for key, value in values.items():
            if isinstance(value, tuple):
                img, caption = value
            else:
                img, caption = (value, '')
            aim_image = aim.Image(img, caption=caption, **aim_image_kw)
            self.writer.track(aim_image, name=key, step=step, **track_kw)

    @on_main_process
    def finish(self):
        """
        Closes `aim` writer
        """
        self.writer.close()