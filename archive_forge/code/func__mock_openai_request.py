import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
def _mock_openai_request():
    original = requests.post

    def request(*args, **kwargs):
        url = kwargs.get('url')
        for key in kwargs.get('json'):
            assert key in REQUEST_FIELDS, f"'{key}' is not a valid request field"
        if '/chat/completions' in url:
            messages = kwargs.get('json').get('messages')
            return _mock_chat_completion_response(content=json.dumps(messages))
        elif '/completions' in url:
            prompt = kwargs.get('json').get('prompt')
            return _mock_completion_response(content=json.dumps(prompt))
        elif '/embeddings' in url:
            inp = kwargs.get('json').get('input')
            return _mock_embeddings_response(len(inp) if isinstance(inp, list) else 1)
        else:
            return original(*args, **kwargs)
    return _mock_request_post(new=request)