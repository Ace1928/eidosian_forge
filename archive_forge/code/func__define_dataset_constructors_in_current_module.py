import sys
from contextlib import suppress
from typing import Union
from mlflow.data import dataset_registry
from mlflow.data import sources as mlflow_data_sources
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import get_dataset_source_from_json, get_registered_sources
from mlflow.entities import Dataset as DatasetEntity
from mlflow.entities import DatasetInput
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _define_dataset_constructors_in_current_module():
    data_module = sys.modules[__name__]
    for constructor_name, constructor_fn in dataset_registry.get_registered_constructors().items():
        setattr(data_module, constructor_name, constructor_fn)
        __all__.append(constructor_name)