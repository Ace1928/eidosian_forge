import json
import logging
import numpy as np
from mlflow.environment_variables import MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import _contains_params
from mlflow.types.schema import ColSpec, DataType, Schema, TensorSpec
from mlflow.utils.os import is_windows
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout
def _infer_signature_with_example(pipeline, example, model_config=None, flavor_config=None, timeout=None) -> ModelSignature:
    params = None
    if _contains_params(example):
        example, params = example
    example = format_input_example_for_special_cases(example, pipeline)
    if timeout:
        _logger.info(f'Running model prediction to infer the model output signature with a timeout of {timeout} seconds. You can specify a different timeout by setting the environment variable {MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT}.')
        with run_with_timeout(timeout):
            prediction = generate_signature_output(pipeline, example, model_config, flavor_config, params)
    else:
        prediction = generate_signature_output(pipeline, example, model_config, flavor_config, params)
    return infer_signature(example, prediction, params)