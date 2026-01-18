import paddle
import mlflow
from mlflow.utils.autologging_utils import (

    Logs early stopping behavior results (e.g. stopped epoch) as metrics to MLflow.

    Args:
        early_stop_callback: The early stopping callback instance used during training.
        client: An `MlflowAutologgingQueueingClient` instance used for MLflow logging.
        run_id: The ID of the MLflow Run to which to log configuration parameters.
    