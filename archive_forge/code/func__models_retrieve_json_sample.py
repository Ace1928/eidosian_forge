import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
def _models_retrieve_json_sample():
    return {'id': 'gpt-3.5-turbo', 'object': 'model', 'owned_by': 'openai', 'permission': []}