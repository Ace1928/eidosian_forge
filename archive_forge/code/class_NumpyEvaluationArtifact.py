import json
import pathlib
import pickle
from collections import namedtuple
from json import JSONDecodeError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationArtifact
from mlflow.utils.annotations import developer_stable
from mlflow.utils.proto_json_utils import NumpyEncoder
@developer_stable
class NumpyEvaluationArtifact(EvaluationArtifact):

    def _save(self, output_artifact_path):
        np.save(output_artifact_path, self._content, allow_pickle=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = np.load(local_artifact_path, allow_pickle=False)
        return self._content