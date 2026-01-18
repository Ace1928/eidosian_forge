import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
class MLflowTracker(GeneralTracker):
    """
    A `Tracker` class that supports `mlflow`. Should be initialized at the start of your script.

    Args:
        experiment_name (`str`, *optional*):
            Name of the experiment. Environment variable MLFLOW_EXPERIMENT_NAME has priority over this argument.
        logging_dir (`str` or `os.PathLike`, defaults to `"."`):
            Location for mlflow logs to be stored.
        run_id (`str`, *optional*):
            If specified, get the run with the specified UUID and log parameters and metrics under that run. The run’s
            end time is unset and its status is set to running, but the run’s other attributes (source_version,
            source_type, etc.) are not changed. Environment variable MLFLOW_RUN_ID has priority over this argument.
        tags (`Dict[str, str]`, *optional*):
            An optional `dict` of `str` keys and values, or a `str` dump from a `dict`, to set as tags on the run. If a
            run is being resumed, these tags are set on the resumed run. If a new run is being created, these tags are
            set on the new run. Environment variable MLFLOW_TAGS has priority over this argument.
        nested_run (`bool`, *optional*, defaults to `False`):
            Controls whether run is nested in parent run. True creates a nested run. Environment variable
            MLFLOW_NESTED_RUN has priority over this argument.
        run_name (`str`, *optional*):
            Name of new run (stored as a mlflow.runName tag). Used only when `run_id` is unspecified.
        description (`str`, *optional*):
            An optional string that populates the description box of the run. If a run is being resumed, the
            description is set on the resumed run. If a new run is being created, the description is set on the new
            run.
    """
    name = 'mlflow'
    requires_logging_directory = False

    @on_main_process
    def __init__(self, experiment_name: str=None, logging_dir: Optional[Union[str, os.PathLike]]=None, run_id: Optional[str]=None, tags: Optional[Union[Dict[str, Any], str]]=None, nested_run: Optional[bool]=False, run_name: Optional[str]=None, description: Optional[str]=None):
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', experiment_name)
        run_id = os.environ.get('MLFLOW_RUN_ID', run_id)
        tags = os.environ.get('MLFLOW_TAGS', tags)
        if isinstance(tags, str):
            tags = json.loads(tags)
        nested_run = os.environ.get('MLFLOW_NESTED_RUN', nested_run)
        import mlflow
        exps = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")
        if len(exps) > 0:
            if len(exps) > 1:
                logger.warning('Multiple experiments with the same name found. Using first one.')
            experiment_id = exps[0].experiment_id
        else:
            experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=logging_dir, tags=tags)
        self.active_run = mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=nested_run, tags=tags, description=description)
        logger.debug(f'Initialized mlflow experiment {experiment_name}')
        logger.debug('Make sure to log any initial configurations with `self.store_init_configuration` before training!')

    @property
    def tracker(self):
        return self.active_run

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs.
        """
        import mlflow
        for name, value in list(values.items()):
            if len(str(value)) > mlflow.utils.validation.MAX_PARAM_VAL_LENGTH:
                logger.warning_once(f'''Accelerate is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow's log_param() only accepts values no longer than {mlflow.utils.validation.MAX_PARAM_VAL_LENGTH} characters so we dropped this attribute.''')
                del values[name]
        values_list = list(values.items())
        for i in range(0, len(values_list), mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH):
            mlflow.log_params(dict(values_list[i:i + mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH]))
        logger.debug('Stored initial configuration hyperparameters to MLflow')

    @on_main_process
    def log(self, values: dict, step: Optional[int]):
        """
        Logs `values` to the current run.

        Args:
            values (`dict`):
                Values to be logged as key-value pairs.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        metrics = {}
        for k, v in values.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            else:
                logger.warning_once(f'''MLflowTracker is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. MLflow's log_metric() only accepts float and int types so we dropped this attribute.''')
        import mlflow
        mlflow.log_metrics(metrics, step=step)
        logger.debug('Successfully logged to mlflow')

    @on_main_process
    def finish(self):
        """
        End the active MLflow run.
        """
        import mlflow
        mlflow.end_run()