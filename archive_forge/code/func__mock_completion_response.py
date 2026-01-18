import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
def _mock_completion_response(content=TEST_CONTENT):
    return _MockResponse(200, _completion_json_sample(content))