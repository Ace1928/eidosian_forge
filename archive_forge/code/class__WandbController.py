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
class _WandbController:
    """Sweep controller class.

    Internal datastructures on the sweep object to coordinate local controller with
    cloud controller.

    Data structures:
        controller: {
            schedule: [
                { id: SCHEDULE_ID
                  data: {param1: val1, param2: val2}},
            ]
            earlystop: [RUN_ID, ...]
        scheduler:
            scheduled: [
                { id: SCHEDULE_ID
                  runid: RUN_ID},
            ]

    `controller` is only updated by the client
    `scheduler` is only updated by the cloud backend

    Protocols:
        Scheduling a run:
        - client controller adds a schedule entry on the controller.schedule list
        - cloud backend notices the new entry and creates a run with the parameters
        - cloud backend adds a scheduled entry on the scheduler.scheduled list
        - client controller notices that the run has been scheduled and removes it from
          controller.schedule list

    Current implementation details:
        - Runs are only schedule if there are no other runs scheduled.

    """

    def __init__(self, sweep_id_or_config=None, entity=None, project=None):
        self._sweep_id: Optional[str] = None
        self._create: Dict = {}
        self._custom_search: Optional[Callable[[Union[dict, sweeps.SweepConfig], List[sweeps.SweepRun]], Optional[sweeps.SweepRun]]] = None
        self._custom_stopping: Optional[Callable[[Union[dict, sweeps.SweepConfig], List[sweeps.SweepRun]], List[sweeps.SweepRun]]] = None
        self._program_function = None
        self._sweep_obj = None
        self._sweep_config: Optional[Union[dict, sweeps.SweepConfig]] = None
        self._sweep_metric: Optional[str] = None
        self._sweep_runs: Optional[List[sweeps.SweepRun]] = None
        self._sweep_runs_map: Optional[Dict[str, sweeps.SweepRun]] = None
        self._scheduler: Optional[Dict] = None
        self._controller: Optional[Dict] = None
        self._controller_prev_step: Optional[Dict] = None
        self._started: bool = False
        self._done_scheduling: bool = False
        self._defer_sweep_creation: bool = False
        self._logged: int = 0
        self._laststatus: str = ''
        self._log_actions: List[Tuple[str, str]] = []
        self._log_debug: List[str] = []
        environ = os.environ
        if entity:
            env.set_entity(entity, env=environ)
        if project:
            env.set_project(project, env=environ)
        self._api = InternalApi(environ=environ)
        if isinstance(sweep_id_or_config, str):
            self._sweep_id = sweep_id_or_config
        elif isinstance(sweep_id_or_config, dict) or isinstance(sweep_id_or_config, sweeps.SweepConfig):
            self._create = sweeps.SweepConfig(sweep_id_or_config)
            for config_key, controller_attr in zip(['method', 'early_terminate'], ['_custom_search', '_custom_stopping']):
                if callable(config_key in self._create and self._create[config_key]):
                    setattr(self, controller_attr, self._create[config_key])
                    self._create[config_key] = 'custom'
            self._sweep_id = self.create(from_dict=True)
        elif sweep_id_or_config is None:
            self._defer_sweep_creation = True
            return
        else:
            raise ControllerError('Unhandled sweep controller type')
        sweep_obj = self._sweep_object_read_from_backend()
        if sweep_obj is None:
            raise ControllerError('Can not find sweep')
        self._sweep_obj = sweep_obj

    def configure_search(self, search: Union[str, Callable[[Union[dict, sweeps.SweepConfig], List[sweeps.SweepRun]], Optional[sweeps.SweepRun]]]):
        self._configure_check()
        if isinstance(search, str):
            self._create['method'] = search
        elif callable(search):
            self._create['method'] = 'custom'
            self._custom_search = search
        else:
            raise ControllerError('Unhandled search type.')

    def configure_stopping(self, stopping: Union[str, Callable[[Union[dict, sweeps.SweepConfig], List[sweeps.SweepRun]], List[sweeps.SweepRun]]], **kwargs):
        self._configure_check()
        if isinstance(stopping, str):
            self._create.setdefault('early_terminate', {})
            self._create['early_terminate']['type'] = stopping
            for k, v in kwargs.items():
                self._create['early_terminate'][k] = v
        elif callable(stopping):
            self._custom_stopping = stopping(kwargs)
            self._create.setdefault('early_terminate', {})
            self._create['early_terminate']['type'] = 'custom'
        else:
            raise ControllerError('Unhandled stopping type.')

    def configure_metric(self, metric, goal=None):
        self._configure_check()
        self._create.setdefault('metric', {})
        self._create['metric']['name'] = metric
        if goal:
            self._create['metric']['goal'] = goal

    def configure_program(self, program):
        self._configure_check()
        if isinstance(program, str):
            self._create['program'] = program
        elif callable(program):
            self._create['program'] = '__callable__'
            self._program_function = program
            raise ControllerError('Program functions are not supported yet')
        else:
            raise ControllerError('Unhandled sweep program type')

    def configure_name(self, name):
        self._configure_check()
        self._create['name'] = name

    def configure_description(self, description):
        self._configure_check()
        self._create['description'] = description

    def configure_parameter(self, name, values=None, value=None, distribution=None, min=None, max=None, mu=None, sigma=None, q=None, a=None, b=None):
        self._configure_check()
        self._create.setdefault('parameters', {}).setdefault(name, {})
        if value is not None or (values is None and min is None and (max is None) and (distribution is None)):
            self._create['parameters'][name]['value'] = value
        if values is not None:
            self._create['parameters'][name]['values'] = values
        if min is not None:
            self._create['parameters'][name]['min'] = min
        if max is not None:
            self._create['parameters'][name]['max'] = max
        if mu is not None:
            self._create['parameters'][name]['mu'] = mu
        if sigma is not None:
            self._create['parameters'][name]['sigma'] = sigma
        if q is not None:
            self._create['parameters'][name]['q'] = q
        if a is not None:
            self._create['parameters'][name]['a'] = a
        if b is not None:
            self._create['parameters'][name]['b'] = b

    def configure_controller(self, type):
        """Configure controller to local if type == 'local'."""
        self._configure_check()
        self._create.setdefault('controller', {})
        self._create['controller'].setdefault('type', type)

    def configure(self, sweep_dict_or_config):
        self._configure_check()
        if self._create:
            raise ControllerError('Already configured.')
        if isinstance(sweep_dict_or_config, dict):
            self._create = sweep_dict_or_config
        elif isinstance(sweep_dict_or_config, str):
            self._create = yaml.safe_load(sweep_dict_or_config)
        else:
            raise ControllerError('Unhandled sweep controller type')

    @property
    def sweep_config(self) -> Union[dict, sweeps.SweepConfig]:
        return self._sweep_config

    @property
    def sweep_id(self) -> str:
        return self._sweep_id

    def _log(self) -> None:
        self._logged += 1

    def _error(self, s: str) -> None:
        print('ERROR:', s)
        self._log()

    def _warn(self, s: str) -> None:
        print('WARN:', s)
        self._log()

    def _info(self, s: str) -> None:
        print('INFO:', s)
        self._log()

    def _debug(self, s: str) -> None:
        print('DEBUG:', s)
        self._log()

    def _configure_check(self) -> None:
        if self._started:
            raise ControllerError('Can not configure after sweep has been started.')

    def _validate(self, config: Dict) -> str:
        violations = sweeps.schema_violations_from_proposed_config(config)
        msg = sweep_config_err_text_from_jsonschema_violations(violations) if len(violations) > 0 else ''
        return msg

    def create(self, from_dict: bool=False) -> str:
        if self._started:
            raise ControllerError('Can not create after sweep has been started.')
        if not self._defer_sweep_creation and (not from_dict):
            raise ControllerError('Can not use create on already created sweep.')
        if not self._create:
            raise ControllerError('Must configure sweep before create.')
        self._create = sweeps.SweepConfig(self._create)
        sweep_id, warnings = self._api.upsert_sweep(self._create)
        handle_sweep_config_violations(warnings)
        print('Create sweep with ID:', sweep_id)
        sweep_url = wandb_sweep._get_sweep_url(self._api, sweep_id)
        if sweep_url:
            print('Sweep URL:', sweep_url)
        self._sweep_id = sweep_id
        self._defer_sweep_creation = False
        return sweep_id

    def run(self, verbose: bool=False, print_status: bool=True, print_actions: bool=False, print_debug: bool=False) -> None:
        if verbose:
            print_status = True
            print_actions = True
            print_debug = True
        self._start_if_not_started()
        while not self.done():
            if print_status:
                self.print_status()
            self.step()
            if print_actions:
                self.print_actions()
            if print_debug:
                self.print_debug()
            time.sleep(5)

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

    def _sweep_object_sync_to_backend(self) -> None:
        if self._controller == self._controller_prev_step:
            return
        sweep_obj_id = self._sweep_obj['id']
        controller = json.dumps(self._controller)
        _, warnings = self._api.upsert_sweep(self._sweep_config, controller=controller, obj_id=sweep_obj_id)
        handle_sweep_config_violations(warnings)
        self._controller_prev_step = self._controller.copy()

    def _start_if_not_started(self) -> None:
        if self._started:
            return
        if self._defer_sweep_creation:
            raise ControllerError('Must specify or create a sweep before running controller.')
        obj = self._sweep_object_read_from_backend()
        if not obj:
            return
        is_local = self._sweep_config.get('controller', {}).get('type') == 'local'
        if not is_local:
            raise ControllerError('Only sweeps with a local controller are currently supported.')
        self._started = True
        self._controller = {}
        self._sweep_object_sync_to_backend()

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

    def _step(self) -> None:
        self._start_if_not_started()
        self._sweep_object_read_from_backend()
        started_ids, stopped_runs, done_runs = self._parse_scheduled()
        schedule_list = self._controller.get('schedule', [])
        new_schedule_list = [s for s in schedule_list if s.get('id') not in started_ids]
        self._controller['schedule'] = new_schedule_list
        earlystop_list = self._controller.get('earlystop', [])
        new_earlystop_list = [r for r in earlystop_list if r not in stopped_runs and r not in done_runs]
        self._controller['earlystop'] = new_earlystop_list
        self._log_actions = []
        self._log_debug = []

    def step(self) -> None:
        self._step()
        suggestion = self.search()
        self.schedule(suggestion)
        to_stop = self.stopping()
        if len(to_stop) > 0:
            self.stop_runs(to_stop)

    def done(self) -> bool:
        self._start_if_not_started()
        state = self._sweep_obj.get('state')
        if state in [s.upper() for s in (sweeps.RunState.preempting.value, SWEEP_INITIAL_RUN_STATE.value, sweeps.RunState.running.value)]:
            return False
        return True

    def _search(self) -> Optional[sweeps.SweepRun]:
        search = self._custom_search or sweeps.next_run
        next_run = search(self._sweep_config, self._sweep_runs or [])
        if next_run is None:
            self._done_scheduling = True
        return next_run

    def search(self) -> Optional[sweeps.SweepRun]:
        self._start_if_not_started()
        suggestion = self._search()
        return suggestion

    def _stopping(self) -> List[sweeps.SweepRun]:
        if 'early_terminate' not in self.sweep_config:
            return []
        stopper = self._custom_stopping or sweeps.stop_runs
        stop_runs = stopper(self._sweep_config, self._sweep_runs or [])
        debug_lines = '\n'.join([' '.join([f'{k}={v}' for k, v in run.early_terminate_info.items()]) for run in stop_runs if run.early_terminate_info is not None])
        if debug_lines:
            self._log_debug += debug_lines
        return stop_runs

    def stopping(self) -> List[sweeps.SweepRun]:
        self._start_if_not_started()
        return self._stopping()

    def schedule(self, run: Optional[sweeps.SweepRun]) -> None:
        self._start_if_not_started()
        if self._controller and self._controller.get('schedule'):
            return
        schedule_id = _id_generator()
        if run is None:
            schedule_list = [{'id': schedule_id, 'data': {'args': None}}]
        else:
            param_list = ['{}={}'.format(k, v.get('value')) for k, v in sorted(run.config.items())]
            self._log_actions.append(('schedule', ','.join(param_list)))
            schedule_list = [{'id': schedule_id, 'data': {'args': run.config}}]
        self._controller['schedule'] = schedule_list
        self._sweep_object_sync_to_backend()

    def stop_runs(self, runs: List[sweeps.SweepRun]) -> None:
        earlystop_list = list({run.name for run in runs})
        self._log_actions.append(('stop', ','.join(earlystop_list)))
        self._controller['earlystop'] = earlystop_list
        self._sweep_object_sync_to_backend()

    def print_status(self) -> None:
        status = _sweep_status(self._sweep_obj, self._sweep_config, self._sweep_runs)
        if self._laststatus != status or self._logged:
            print(status)
        self._laststatus = status
        self._logged = 0

    def print_actions(self) -> None:
        for action, line in self._log_actions:
            self._info(f'{action.capitalize()} ({line})')
        self._log_actions = []

    def print_debug(self) -> None:
        for line in self._log_debug:
            self._debug(line)
        self._log_debug = []

    def print_space(self) -> None:
        self._warn('Method not implemented yet.')

    def print_summary(self) -> None:
        self._warn('Method not implemented yet.')