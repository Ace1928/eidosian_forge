import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
def _completion_json_sample(content):
    return {'id': 'cmpl-123', 'object': 'text_completion', 'created': 1589478378, 'model': 'text-davinci-003', 'choices': [{'text': content, 'index': 0, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12}}