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
Sync fine-tunes to Weights & Biases.

        :param fine_tune_job_id: The id of the fine-tune (optional)
        :param openai_client: Pass the `OpenAI()` client (optional)
        :param num_fine_tunes: Number of most recent fine-tunes to log when an fine_tune_job_id is not provided. By default, every fine-tune is synced.
        :param project: Name of the project where you're sending runs. By default, it is "GPT-3".
        :param entity: Username or team name where you're sending runs. By default, your default entity is used, which is usually your username.
        :param overwrite: Forces logging and overwrite existing wandb run of the same fine-tune.
        :param wait_for_job_success: Waits for the fine-tune to be complete and then log metrics to W&B. By default, it is True.
        