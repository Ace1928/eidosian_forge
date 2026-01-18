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
def _try_load_executable(self) -> bool:
    """Check existance of valid executable for a run.

        logs and returns False when job is unreachable
        """
    if self._kwargs.get('job'):
        try:
            _job_artifact = self._public_api.job(self._kwargs['job'])
            wandb.termlog(f'{LOG_PREFIX}Successfully loaded job ({_job_artifact.name}) in scheduler')
        except Exception:
            wandb.termerror(f'{LOG_PREFIX}{traceback.format_exc()}')
            return False
        return True
    elif self._kwargs.get('image_uri'):
        return True
    else:
        return False