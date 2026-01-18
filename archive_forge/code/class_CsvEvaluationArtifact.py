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
class CsvEvaluationArtifact(EvaluationArtifact):

    def _save(self, output_artifact_path):
        self._content.to_csv(output_artifact_path, index=False)

    def _load_content_from_file(self, local_artifact_path):
        self._content = pd.read_csv(local_artifact_path)
        return self._content