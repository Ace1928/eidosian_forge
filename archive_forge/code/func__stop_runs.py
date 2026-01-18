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
def _stop_runs(self) -> None:
    to_delete = []
    for run_id, _ in self._yield_runs():
        to_delete += [run_id]
    for run_id in to_delete:
        wandb.termlog(f'{LOG_PREFIX}Stopping run ({run_id})')
        if not self._stop_run(run_id):
            wandb.termwarn(f'{LOG_PREFIX}Failed to stop run ({run_id})')