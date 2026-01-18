from pathlib import Path
from typing import TYPE_CHECKING, Callable
import lightgbm  # type: ignore
from lightgbm import Booster
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
class CallbackEnv(NamedTuple):
    model: Any
    params: Dict
    iteration: int
    begin_interation: int
    end_iteration: int
    evaluation_result_list: List[_EvalResultTuple]