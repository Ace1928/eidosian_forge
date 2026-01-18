import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from packaging.version import Version
import mlflow
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.autologging_utils import (
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags
def _combine_input_and_output(input, output, session_id, func_name):
    """
    Combine input and output into a single dictionary
    """
    if func_name == 'get_relevant_documents' and output is not None:
        output = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in output]
        output = [output]
    result = {'session_id': [session_id]}
    if input:
        result.update(_convert_data_to_dict(input, 'input'))
    if output:
        result.update(_convert_data_to_dict(output, 'output'))
    return result