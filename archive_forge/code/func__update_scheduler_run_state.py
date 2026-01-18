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
def _update_scheduler_run_state(self) -> None:
    """Update the scheduler state from state of scheduler run and sweep state."""
    state: RunState = self._get_run_state(self._wandb_run.id)
    if state == RunState.KILLED:
        self.state = SchedulerState.STOPPED
    elif state in [RunState.FAILED, RunState.CRASHED]:
        self.state = SchedulerState.FAILED
    elif state == RunState.FINISHED:
        self.state = SchedulerState.COMPLETED
    try:
        sweep_state = self._api.get_sweep_state(self._sweep_id, self._entity, self._project)
    except Exception as e:
        _logger.debug(f'sweep state error: {e}')
        return
    if sweep_state == 'FINISHED':
        self.state = SchedulerState.COMPLETED
    elif sweep_state in ['CANCELLED', 'STOPPED']:
        self.state = SchedulerState.CANCELLED
    elif sweep_state == 'PAUSED':
        self.state = SchedulerState.FLUSH_RUNS