from pathlib import Path
from typing import TYPE_CHECKING, Callable
import lightgbm  # type: ignore
from lightgbm import Booster
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
class _WandbCallback:
    """Internal class to handle `wandb_callback` logic.

    This callback is adapted form the LightGBM's `_RecordEvaluationCallback`.
    """

    def __init__(self, log_params: bool=True, define_metric: bool=True) -> None:
        self.order = 20
        self.before_iteration = False
        self.log_params = log_params
        self.define_metric_bool = define_metric

    def _init(self, env: 'CallbackEnv') -> None:
        with wb_telemetry.context() as tel:
            tel.feature.lightgbm_wandb_callback = True
        if self.log_params:
            wandb.config.update(env.params)
        for item in env.evaluation_result_list:
            if self.define_metric_bool:
                if len(item) == 4:
                    data_name, eval_name = item[:2]
                    _define_metric(data_name, eval_name)
                else:
                    data_name, eval_name = item[1].split()
                    _define_metric(data_name, f'{eval_name}-mean')
                    _define_metric(data_name, f'{eval_name}-stdv')

    def __call__(self, env: 'CallbackEnv') -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        for item in env.evaluation_result_list:
            if len(item) == 4:
                data_name, eval_name, result = item[:3]
                wandb.log({data_name + '_' + eval_name: result}, commit=False)
            else:
                data_name, eval_name = item[1].split()
                res_mean = item[2]
                res_stdv = item[4]
                wandb.log({data_name + '_' + eval_name + '-mean': res_mean, data_name + '_' + eval_name + '-stdv': res_stdv}, commit=False)
        wandb.log({'iteration': env.iteration}, commit=True)