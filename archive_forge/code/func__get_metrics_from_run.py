import asyncio
import base64
import copy
import logging
import os
import socket
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union
import click
import yaml
import wandb
from wandb.errors import CommError
from wandb.sdk.launch._launch_add import launch_add
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.sweeps import SchedulerError
from wandb.sdk.launch.sweeps.utils import (
from wandb.sdk.launch.utils import (
from wandb.sdk.lib.runid import generate_id
def _get_metrics_from_run(self, run_id: str) -> List[Any]:
    """Use the public api to get metrics from a run.

        Uses the metric name found in the sweep config, any
        misspellings will result in an empty list.
        """
    try:
        queued_run: Optional[QueuedRun] = self._runs[run_id].queued_run
        if not queued_run:
            return []
        api_run: Run = self._public_api.run(f'{queued_run.entity}/{queued_run.project}/{run_id}')
        metric_name = self._sweep_config['metric']['name']
        history = api_run.scan_history(keys=['_step', metric_name])
        metrics = [x[metric_name] for x in history]
        return metrics
    except Exception as e:
        _logger.debug(f'[_get_metrics_from_run] {e}')
    return []