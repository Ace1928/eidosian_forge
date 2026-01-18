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
def _get_run_info(self, run_id: str) -> Dict[str, Any]:
    """Use the public api to get info about a run."""
    try:
        info: Dict[str, Any] = self._api.get_run_info(self._entity, self._project, run_id)
        if info:
            return info
    except Exception as e:
        _logger.debug(f'[_get_run_info] {e}')
    return {}