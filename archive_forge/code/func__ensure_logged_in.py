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
def _ensure_logged_in(cls):
    if not cls._logged_in:
        if wandb.login():
            cls._logged_in = True
        else:
            raise Exception('It appears you are not currently logged in to Weights & Biases. Please run `wandb login` in your terminal. When prompted, you can obtain your API key by visiting wandb.ai/authorize.')