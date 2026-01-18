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
def _get_run_state(self, run_id: str, prev_run_state: RunState=RunState.UNKNOWN) -> RunState:
    """Use the public api to get state of a run."""
    run_state = None
    try:
        state = self._api.get_run_state(self._entity, self._project, run_id)
        run_state = RunState(state)
    except CommError as e:
        _logger.debug(f'error getting state for run ({run_id}): {e}')
        if prev_run_state == RunState.UNKNOWN:
            wandb.termwarn(f'Failed to get runstate for run ({run_id}). Error: {traceback.format_exc()}')
            run_state = RunState.FAILED
        else:
            run_state = RunState.UNKNOWN
    except (AttributeError, ValueError):
        wandb.termwarn(f'Bad state ({run_state}) for run ({run_id}). Error: {traceback.format_exc()}')
        run_state = RunState.UNKNOWN
    return run_state