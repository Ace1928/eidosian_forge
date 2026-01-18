import logging
import os
import re
import signal
import sys
import threading
from subprocess import call
from types import FrameType
from typing import Any, Callable, Dict, List, Set, Union
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS, _PYTHON_GREATER_EQUAL_3_8_0
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_info
def _slurm_sigusr_handler_fn(self, signum: _SIGNUM, _: FrameType) -> None:
    rank_zero_info(f'Handling auto-requeue signal: {signum}')
    for logger in self.trainer.loggers:
        logger.finalize('finished')
    hpc_save_path = self.trainer._checkpoint_connector.hpc_save_path(self.trainer.default_root_dir)
    self.trainer.save_checkpoint(hpc_save_path)
    if self.trainer.is_global_zero:
        array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
        if array_job_id is not None:
            array_task_id = os.environ['SLURM_ARRAY_TASK_ID']
            job_id = f'{array_job_id}_{array_task_id}'
        else:
            job_id = os.environ['SLURM_JOB_ID']
        assert re.match('[0-9_-]+', job_id)
        cmd = ['scontrol', 'requeue', job_id]
        log.info(f'requeing job {job_id}...')
        try:
            result = call(cmd)
        except FileNotFoundError:
            result = call(' '.join(cmd), shell=True)
        if result == 0:
            log.info(f'Requeued SLURM job: {job_id}')
        else:
            log.warning(f'Requeuing SLURM job {job_id} failed with error code {result}')