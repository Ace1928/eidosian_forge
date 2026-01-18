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
def _add_to_launch_queue(self, run: SweepRun) -> bool:
    """Convert a sweeprun into a launch job then push to runqueue."""
    _job = self._kwargs.get('job') or self._sweep_config.get('job')
    _sweep_config_uri = self._sweep_config.get('image_uri')
    _image_uri = self._kwargs.get('image_uri') or _sweep_config_uri
    if _job is None and _image_uri is None:
        raise SchedulerError(f"{LOG_PREFIX}No 'job' nor 'image_uri' ({run.id})")
    elif _job is not None and _image_uri is not None:
        raise SchedulerError(f"{LOG_PREFIX}Sweep has both 'job' and 'image_uri'")
    entry_point, launch_config = self._make_entry_and_launch_config(run)
    if entry_point:
        wandb.termwarn(f'{LOG_PREFIX}Sweep command {entry_point} will override {('job' if _job else 'image_uri')} entrypoint')
    _job_launch_config = copy.deepcopy(self._wandb_run.config.get('launch')) or {}
    _priority = int(launch_config.get('priority', 2))
    strip_resource_args_and_template_vars(_job_launch_config)
    run_id = run.id or generate_id()
    queued_run = launch_add(run_id=run_id, entry_point=entry_point, config=launch_config, docker_image=_image_uri, job=_job, project=self._project, entity=self._entity, queue_name=self._kwargs.get('queue'), project_queue=self._project_queue, resource=_job_launch_config.get('resource'), resource_args=_job_launch_config.get('resource_args'), template_variables=_job_launch_config.get('template_variables'), author=self._kwargs.get('author'), sweep_id=self._sweep_id, priority=_priority)
    run.queued_run = queued_run
    run.state = RunState.RUNNING
    self._runs[run_id] = run
    wandb.termlog(f'{LOG_PREFIX}Added run ({run_id}) to queue ({self._kwargs.get('queue')})')
    return True