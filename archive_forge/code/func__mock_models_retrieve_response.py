import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
def _mock_models_retrieve_response():
    return _MockResponse(200, _models_retrieve_json_sample())