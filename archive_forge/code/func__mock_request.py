import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
@contextmanager
def _mock_request(**kwargs):
    with mock.patch('requests.Session.request', **kwargs) as m:
        yield m