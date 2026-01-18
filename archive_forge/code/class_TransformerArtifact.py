import json
import logging
import os
from abc import ABC, abstractmethod
import mlflow
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.tracking import MlflowClient
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.file_utils import chdir
class TransformerArtifact(Artifact):

    def __init__(self, name, recipe_root, step_name, tracking_uri):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, 'transformer.pkl')
        self._recipe_root = recipe_root
        self._step_name = step_name
        self._tracking_uri = tracking_uri

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        run_id = read_run_id(self._recipe_root)
        if run_id:
            with _use_tracking_uri(self._tracking_uri), chdir(self._recipe_root):
                return mlflow.sklearn.load_model(f'runs:/{run_id}/{self._step_name}/transformer')
        log_artifact_not_found_warning(self._name, self._step_name)
        return None