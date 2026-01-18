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
def _make_entry_and_launch_config(self, run: SweepRun) -> Tuple[Optional[List[str]], Dict[str, Dict[str, Any]]]:
    args = create_sweep_command_args({'args': run.args})
    entry_point, macro_args = make_launch_sweep_entrypoint(args, self._sweep_config.get('command'))
    if entry_point and '${program}' in entry_point:
        if not self._sweep_config.get('program'):
            raise SchedulerError(f"{LOG_PREFIX}Program macro in command has no corresponding 'program' in sweep config.")
        pidx = entry_point.index('${program}')
        entry_point[pidx] = self._sweep_config['program']
    launch_config = copy.deepcopy(self._wandb_run.config.get('launch', {}))
    if 'overrides' not in launch_config:
        launch_config['overrides'] = {'run_config': {}}
    launch_config['overrides']['run_config'].update(args['args_dict'])
    if macro_args:
        launch_config['overrides']['args'] = macro_args
    if entry_point:
        unresolved = [x for x in entry_point if str(x).startswith('${')]
        if unresolved:
            wandb.termwarn(f'{LOG_PREFIX}Sweep command contains unresolved macros: {unresolved}, see launch docs for supported macros.')
    return (entry_point, launch_config)