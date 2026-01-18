import warnings
import click
from mlflow.environment_variables import MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING
from mlflow.utils import env_manager as _EnvManager
def _create_env_manager_option(help_string, default=None):
    return click.option('--env-manager', default=default, type=click.UNPROCESSED, callback=_resolve_env_manager, help=help_string)