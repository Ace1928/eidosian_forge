from pathlib import Path
from typing import TYPE_CHECKING, Callable
import lightgbm  # type: ignore
from lightgbm import Booster
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def _define_metric(data: str, metric_name: str) -> None:
    """Capture model performance at the best step.

    instead of the last step, of training in your `wandb.summary`
    """
    if 'loss' in str.lower(metric_name):
        wandb.define_metric(f'{data}_{metric_name}', summary='min')
    elif str.lower(metric_name) in MINIMIZE_METRICS:
        wandb.define_metric(f'{data}_{metric_name}', summary='min')
    elif str.lower(metric_name) in MAXIMIZE_METRICS:
        wandb.define_metric(f'{data}_{metric_name}', summary='max')