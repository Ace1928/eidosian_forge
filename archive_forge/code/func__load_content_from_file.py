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
def _load_content_from_file(self, local_artifact_path):
    with open(local_artifact_path, 'rb') as f:
        self._content = pickle.load(f)
    return self._content