import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
class _OpenAIApiConfig(NamedTuple):
    api_type: str
    batch_size: int
    max_requests_per_minute: int
    max_tokens_per_minute: int
    api_version: Optional[str]
    api_base: str
    engine: Optional[str]
    deployment_id: Optional[str]