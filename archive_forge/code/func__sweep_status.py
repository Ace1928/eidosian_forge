import json
import os
import random
import string
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import yaml
from wandb import env
from wandb.apis import InternalApi
from wandb.sdk import wandb_sweep
from wandb.sdk.launch.sweeps.utils import (
from wandb.util import get_module
def _sweep_status(sweep_obj: dict, sweep_conf: Union[dict, sweeps.SweepConfig], sweep_runs: List[sweeps.SweepRun]) -> str:
    sweep = sweep_obj['name']
    _ = sweep_obj['state']
    run_count = len(sweep_runs)
    run_type_counts = _get_run_counts(sweep_runs)
    stopped = len([r for r in sweep_runs if r.stopped])
    stopping = len([r for r in sweep_runs if r.should_stop])
    stopstr = ''
    if stopped or stopping:
        stopstr = 'Stopped: %d' % stopped
        if stopping:
            stopstr += ' (Stopping: %d)' % stopping
    runs_status = _get_runs_status(run_type_counts)
    method = sweep_conf.get('method', 'unknown')
    stopping = sweep_conf.get('early_terminate', None)
    sweep_options = []
    sweep_options.append(method)
    if stopping:
        sweep_options.append(stopping.get('type', 'unknown'))
    sweep_options = ','.join(sweep_options)
    sections = []
    sections.append(f'Sweep: {sweep} ({sweep_options})')
    if runs_status:
        sections.append('Runs: %d (%s)' % (run_count, runs_status))
    else:
        sections.append('Runs: %d' % run_count)
    if stopstr:
        sections.append(stopstr)
    sections = ' | '.join(sections)
    return sections