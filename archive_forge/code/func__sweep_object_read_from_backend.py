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
def _sweep_object_read_from_backend(self) -> Optional[dict]:
    specs_json = {}
    if self._sweep_metric:
        k = ['_step']
        k.append(self._sweep_metric)
        specs_json = {'keys': k, 'samples': 100000}
    specs = json.dumps(specs_json)
    sweep_obj = self._api.sweep(self._sweep_id, specs)
    if not sweep_obj:
        return
    self._sweep_obj = sweep_obj
    self._sweep_config = yaml.safe_load(sweep_obj['config'])
    self._sweep_metric = self._sweep_config.get('metric', {}).get('name')
    _sweep_runs: List[sweeps.SweepRun] = []
    for r in sweep_obj['runs']:
        rr = r.copy()
        if 'summaryMetrics' in rr:
            if rr['summaryMetrics']:
                rr['summaryMetrics'] = json.loads(rr['summaryMetrics'])
        if 'config' not in rr:
            raise ValueError('sweep object is missing config')
        rr['config'] = json.loads(rr['config'])
        if 'history' in rr:
            if isinstance(rr['history'], list):
                rr['history'] = [json.loads(d) for d in rr['history']]
            else:
                raise ValueError('Invalid history value: expected list of json strings: %s' % rr['history'])
        if 'sampledHistory' in rr:
            sampled_history = []
            for historyDictList in rr['sampledHistory']:
                sampled_history += historyDictList
            rr['sampledHistory'] = sampled_history
        _sweep_runs.append(sweeps.SweepRun(**rr))
    self._sweep_runs = _sweep_runs
    self._sweep_runs_map = {r.name: r for r in self._sweep_runs}
    self._controller = json.loads(sweep_obj.get('controller') or '{}')
    self._scheduler = json.loads(sweep_obj.get('scheduler') or '{}')
    self._controller_prev_step = self._controller.copy()
    return sweep_obj