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
def _set_sweep_state(self, state: str) -> None:
    wandb.termlog(f'{LOG_PREFIX}Updating sweep state to: {state.lower()}')
    try:
        self._api.set_sweep_state(sweep=self._sweep_id, state=state)
    except Exception as e:
        _logger.debug(f'[set_sweep_state] {e}')