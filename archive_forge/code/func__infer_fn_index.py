from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal
import httpx
import huggingface_hub
from huggingface_hub import CommitOperationAdd, SpaceHardware, SpaceStage
from huggingface_hub.utils import (
from packaging import version
from gradio_client import utils
from gradio_client.compatibility import EndpointV3Compatibility
from gradio_client.data_classes import ParameterInfo
from gradio_client.documentation import document
from gradio_client.exceptions import AuthenticationError
from gradio_client.utils import (
def _infer_fn_index(self, api_name: str | None, fn_index: int | None) -> int:
    inferred_fn_index = None
    if api_name is not None:
        for i, d in enumerate(self.config['dependencies']):
            config_api_name = d.get('api_name')
            if config_api_name is None or config_api_name is False:
                continue
            if '/' + config_api_name == api_name:
                inferred_fn_index = i
                break
        else:
            error_message = f'Cannot find a function with `api_name`: {api_name}.'
            if not api_name.startswith('/'):
                error_message += ' Did you mean to use a leading slash?'
            raise ValueError(error_message)
    elif fn_index is not None:
        inferred_fn_index = fn_index
        if inferred_fn_index >= len(self.endpoints) or not self.endpoints[inferred_fn_index].is_valid:
            raise ValueError(f'Invalid function index: {fn_index}.')
    else:
        valid_endpoints = [e for e in self.endpoints if e.is_valid and e.api_name is not None and (e.backend_fn is not None) and e.show_api]
        if len(valid_endpoints) == 1:
            inferred_fn_index = valid_endpoints[0].fn_index
        else:
            raise ValueError('This Gradio app might have multiple endpoints. Please specify an `api_name` or `fn_index`')
    return inferred_fn_index