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
def _cleanup_runs(self, runs_to_remove: List[str]) -> None:
    """Helper for removing runs from memory.

        Can be overloaded to prevent deletion of runs, which is useful
        for debugging or when polling on completed runs.
        """
    with self._threading_lock:
        for run_id in runs_to_remove:
            wandb.termlog(f'{LOG_PREFIX}Cleaning up finished run ({run_id})')
            del self._runs[run_id]