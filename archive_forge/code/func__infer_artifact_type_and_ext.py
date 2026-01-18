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
def _infer_artifact_type_and_ext(artifact_name, raw_artifact, custom_metric_tuple):
    """
    This function performs type and file extension inference on the provided artifact

    Args:
        artifact_name: The name of the provided artifact
        raw_artifact: The artifact object
        custom_metric_tuple: Containing a user provided function and its index in the
            ``custom_metrics`` parameter of ``mlflow.evaluate``

    Returns:
        InferredArtifactProperties namedtuple
    """
    exception_header = f"Custom metric function '{custom_metric_tuple.name}' at index {custom_metric_tuple.index} in the `custom_metrics` parameter produced an artifact '{artifact_name}'"
    if isinstance(raw_artifact, str):
        potential_path = pathlib.Path(raw_artifact)
        if potential_path.exists():
            raw_artifact = potential_path
        else:
            try:
                json.loads(raw_artifact)
                return _InferredArtifactProperties(from_path=False, type=JsonEvaluationArtifact, ext='.json')
            except JSONDecodeError:
                raise MlflowException(f"{exception_header} with string representation '{raw_artifact}' that is neither a valid path to a file nor a JSON string.")
    if isinstance(raw_artifact, pathlib.Path):
        if not raw_artifact.exists():
            raise MlflowException(f"{exception_header} with path '{raw_artifact}' does not exist.")
        if not raw_artifact.is_file():
            raise MlflowException(f"{exception_header} with path '{raw_artifact}' is not a file.")
        if raw_artifact.suffix not in _EXT_TO_ARTIFACT_MAP:
            raise MlflowException(f"{exception_header} with path '{raw_artifact}' does not match any of the supported file extensions: {', '.join(_EXT_TO_ARTIFACT_MAP.keys())}.")
        return _InferredArtifactProperties(from_path=True, type=_EXT_TO_ARTIFACT_MAP[raw_artifact.suffix], ext=raw_artifact.suffix)
    if type(raw_artifact) in _TYPE_TO_ARTIFACT_MAP:
        return _InferredArtifactProperties(from_path=False, type=_TYPE_TO_ARTIFACT_MAP[type(raw_artifact)], ext=_TYPE_TO_EXT_MAP[type(raw_artifact)])
    try:
        json.dumps(raw_artifact, cls=NumpyEncoder)
        return _InferredArtifactProperties(from_path=False, type=JsonEvaluationArtifact, ext='.json')
    except TypeError:
        return _InferredArtifactProperties(from_path=False, type=PickleEvaluationArtifact, ext='.pickle')