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
def _log_artifacts(cls, fine_tune: FineTuningJob, project: str, entity: Optional[str]) -> None:
    training_file = fine_tune.training_file if fine_tune.training_file else None
    validation_file = fine_tune.validation_file if fine_tune.validation_file else None
    for file, prefix, artifact_type in ((training_file, 'train', 'training_files'), (validation_file, 'valid', 'validation_files')):
        if file is not None:
            cls._log_artifact_inputs(file, prefix, artifact_type, project, entity)
    fine_tune_id = fine_tune.id
    artifact = wandb.Artifact('model_metadata', type='model', metadata=dict(fine_tune))
    with artifact.new_file('model_metadata.json', mode='w', encoding='utf-8') as f:
        dict_fine_tune = dict(fine_tune)
        dict_fine_tune['hyperparameters'] = dict(dict_fine_tune['hyperparameters'])
        json.dump(dict_fine_tune, f, indent=2)
    cls._run.log_artifact(artifact, aliases=['latest', fine_tune_id])