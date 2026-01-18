import ctypes
import logging
import os
import queue
import socket
import threading
import time
import traceback
import wandb
from wandb import wandb_sdk
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps import utils as sweep_utils
def _run_job(self, job):
    try:
        run_id = job.run_id
        config_file = os.path.join('wandb', 'sweep-' + self._sweep_id, 'config-' + run_id + '.yaml')
        os.environ[wandb.env.RUN_ID] = run_id
        base_dir = os.environ.get(wandb.env.DIR, '')
        sweep_param_path = os.path.join(base_dir, config_file)
        os.environ[wandb.env.SWEEP_PARAM_PATH] = sweep_param_path
        wandb.wandb_lib.config_util.save_config_file_from_dict(sweep_param_path, job.config)
        os.environ[wandb.env.SWEEP_ID] = self._sweep_id
        wandb_sdk.wandb_setup._setup(_reset=True)
        wandb.termlog(f'Agent Starting Run: {run_id} with config:')
        for k, v in job.config.items():
            wandb.termlog('\t{}: {}'.format(k, v['value']))
        self._function()
        wandb.finish()
    except KeyboardInterrupt as ki:
        raise ki
    except Exception as e:
        wandb.finish(exit_code=1)
        if self._run_status[run_id] == RunStatus.RUNNING:
            self._run_status[run_id] = RunStatus.ERRORED
            self._exceptions[run_id] = e
    finally:
        os.environ.pop(wandb.env.RUN_ID, None)
        os.environ.pop(wandb.env.SWEEP_ID, None)
        os.environ.pop(wandb.env.SWEEP_PARAM_PATH, None)