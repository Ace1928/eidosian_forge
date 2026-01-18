import time
from contextlib import contextmanager
import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.metrics_queue import (
use a synchronous call here since this is going to get called very infrequently.