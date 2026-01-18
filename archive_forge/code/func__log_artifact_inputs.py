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
def _log_artifact_inputs(cls, file_id: Optional[str], prefix: str, artifact_type: str, project: str, entity: Optional[str]) -> None:
    artifact_name = f'{prefix}-{file_id}'
    artifact_name = re.sub('[^a-zA-Z0-9_\\-.]', '_', artifact_name)
    artifact_alias = file_id
    artifact_path = f'{project}/{artifact_name}:{artifact_alias}'
    if entity is not None:
        artifact_path = f'{entity}/{artifact_path}'
    artifact = cls._get_wandb_artifact(artifact_path)
    if artifact is None:
        try:
            file_content = cls.openai_client.files.retrieve_content(file_id=file_id)
        except openai.NotFoundError:
            wandb.termerror(f'File {file_id} could not be retrieved. Make sure you are allowed to download training/validation files')
            return
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        with artifact.new_file(file_id, mode='w', encoding='utf-8') as f:
            f.write(file_content)
        try:
            table, n_items = cls._make_table(file_content)
            artifact.add(table, file_id)
            cls._run.log({f'{prefix}_data': table})
            cls._run.config.update({f'n_{prefix}': n_items})
            artifact.metadata['items'] = n_items
        except Exception:
            wandb.termerror(f'File {file_id} could not be read as a valid JSON file')
    else:
        cls._run.config.update({f'n_{prefix}': artifact.metadata.get('items')})
    cls._run.use_artifact(artifact, aliases=['latest', artifact_alias])