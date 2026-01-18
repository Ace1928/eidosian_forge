import inspect
import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, get_type_hints
import numpy as np
import pandas as pd
from mlflow import environment_variables
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _contains_params, _Example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.types.schema import ParamSchema, Schema
from mlflow.types.utils import _infer_param_schema, _infer_schema, _infer_schema_from_type_hint
from mlflow.utils.uri import append_to_uri_path
class ModelSignature:
    """
    ModelSignature specifies schema of model's inputs, outputs and params.

    ModelSignature can be :py:func:`inferred <mlflow.models.infer_signature>` from training
    dataset, model predictions using and params for inference, or constructed by hand by
    passing an input and output :py:class:`Schema <mlflow.types.Schema>`, and params
    :py:class:`ParamSchema <mlflow.types.ParamSchema>`.
    """

    def __init__(self, inputs: Schema=None, outputs: Schema=None, params: ParamSchema=None):
        if inputs and (not isinstance(inputs, Schema)):
            raise TypeError(f"inputs must be either None or mlflow.models.signature.Schema, got '{type(inputs).__name__}'")
        if outputs and (not isinstance(outputs, Schema)):
            raise TypeError(f"outputs must be either None or mlflow.models.signature.Schema, got '{type(outputs).__name__}'")
        if params and (not isinstance(params, ParamSchema)):
            raise TypeError(f"If params are provided, they must by of type mlflow.models.signature.ParamSchema, got '{type(params).__name__}'")
        if all((x is None for x in [inputs, outputs, params])):
            raise ValueError('At least one of inputs, outputs or params must be provided')
        self.inputs = inputs
        self.outputs = outputs
        self.params = params

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a 'jsonable' dictionary.

        Input and output schema are represented as json strings. This is so that the
        representation is compact when embedded in an MLmodel yaml file.

        Returns:
            dictionary representation with input and output schema represented as json strings.
        """
        return {'inputs': self.inputs.to_json() if self.inputs else None, 'outputs': self.outputs.to_json() if self.outputs else None, 'params': self.params.to_json() if self.params else None}

    @classmethod
    def from_dict(cls, signature_dict: Dict[str, Any]):
        """
        Deserialize from dictionary representation.

        Args:
            signature_dict: Dictionary representation of model signature.
                Expected dictionary format:
                `{'inputs': <json string>,
                'outputs': <json string>,
                'params': <json string>" }`

        Returns:
            ModelSignature populated with the data form the dictionary.
        """
        inputs = Schema.from_json(x) if (x := signature_dict.get('inputs')) else None
        outputs = Schema.from_json(x) if (x := signature_dict.get('outputs')) else None
        params = ParamSchema.from_json(x) if (x := signature_dict.get('params')) else None
        return cls(inputs, outputs, params)

    def __eq__(self, other) -> bool:
        return isinstance(other, ModelSignature) and self.inputs == other.inputs and (self.outputs == other.outputs) and (self.params == other.params)

    def __repr__(self) -> str:
        return f'inputs: \n  {self.inputs!r}\noutputs: \n  {self.outputs!r}\nparams: \n  {self.params!r}\n'