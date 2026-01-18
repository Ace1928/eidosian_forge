import json
import logging
import os
from abc import ABC, abstractmethod
import mlflow
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.tracking import MlflowClient
from mlflow.tracking._tracking_service.utils import _use_tracking_uri
from mlflow.utils.file_utils import chdir
class HyperParametersArtifact(Artifact):

    def __init__(self, name, recipe_root, step_name):
        self._name = name
        self._path = get_step_output_path(recipe_root, step_name, 'best_parameters.yaml')

    def name(self):
        return self._name

    def path(self):
        return self._path

    def load(self):
        if os.path.exists(self._path):
            with open(self._path) as f:
                return f.read()