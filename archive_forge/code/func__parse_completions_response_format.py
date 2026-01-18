import logging
import os
import urllib.parse
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.openai_utils import REQUEST_URL_CHAT
def _parse_completions_response_format(response):
    try:
        text = response['choices'][0]['text']
    except (KeyError, IndexError, TypeError):
        text = None
    return text