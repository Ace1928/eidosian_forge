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
def _parse_scheduled(self):
    scheduled_list = self._scheduler.get('scheduled') or []
    started_ids = []
    stopped_runs = []
    done_runs = []
    for s in scheduled_list:
        runid = s.get('runid')
        objid = s.get('id')
        r = self._sweep_runs_map.get(runid)
        if not r:
            continue
        if r.stopped:
            stopped_runs.append(runid)
        summary = r.summary_metrics
        if r.state == SWEEP_INITIAL_RUN_STATE and (not summary):
            continue
        started_ids.append(objid)
        if r.state != 'running':
            done_runs.append(runid)
    return (started_ids, stopped_runs, done_runs)