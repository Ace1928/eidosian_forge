import datetime
import io
import json
import re
import time
from typing import Any, Dict, Optional, Tuple
import wandb
from wandb import util
from wandb.data_types import Table
from wandb.sdk.lib import telemetry
from wandb.sdk.wandb_run import Run
from wandb.util import parse_version
from openai import OpenAI  # noqa: E402
from openai.types.fine_tuning import FineTuningJob  # noqa: E402
from openai.types.fine_tuning.fine_tuning_job import Hyperparameters  # noqa: E402
@classmethod
def _log_fine_tune(cls, fine_tune: FineTuningJob, project: str, entity: Optional[str], overwrite: bool, show_individual_warnings: bool, **kwargs_wandb_init: Dict[str, Any]):
    fine_tune_id = fine_tune.id
    status = fine_tune.status
    with telemetry.context(run=cls._run) as tel:
        tel.feature.openai_finetuning = True
    if status != 'succeeded':
        if show_individual_warnings:
            wandb.termwarn(f'Fine-tune {fine_tune_id} has the status "{status}" and will not be logged')
        return
    try:
        results_id = fine_tune.result_files[0]
        results = cls.openai_client.files.retrieve_content(file_id=results_id)
    except openai.NotFoundError:
        if show_individual_warnings:
            wandb.termwarn(f'Fine-tune {fine_tune_id} has no results and will not be logged')
        return
    cls._run.config.update(cls._get_config(fine_tune))
    df_results = pd.read_csv(io.StringIO(results))
    for _, row in df_results.iterrows():
        metrics = {k: v for k, v in row.items() if not np.isnan(v)}
        step = metrics.pop('step')
        if step is not None:
            step = int(step)
        cls._run.log(metrics, step=step)
    fine_tuned_model = fine_tune.fine_tuned_model
    if fine_tuned_model is not None:
        cls._run.summary['fine_tuned_model'] = fine_tuned_model
    cls._log_artifacts(fine_tune, project, entity)
    cls._run.summary['status'] = 'succeeded'
    cls._run.finish()
    return True