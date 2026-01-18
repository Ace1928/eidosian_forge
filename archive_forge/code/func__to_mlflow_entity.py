import json
from abc import abstractmethod
from typing import Any, Dict, Optional
from mlflow.data.dataset_source import DatasetSource
from mlflow.entities import Dataset as DatasetEntity
def _to_mlflow_entity(self) -> DatasetEntity:
    """
        Returns:
            A `mlflow.entities.Dataset` instance representing the dataset.
        """
    dataset_dict = self.to_dict()
    return DatasetEntity(name=dataset_dict['name'], digest=dataset_dict['digest'], source_type=dataset_dict['source_type'], source=dataset_dict['source'], schema=dataset_dict.get('schema'), profile=dataset_dict.get('profile'))