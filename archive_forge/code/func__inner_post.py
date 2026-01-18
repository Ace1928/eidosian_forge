import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
@wraps(FastAPI.post)
def _inner_post(*args, **kwargs):
    func = args[0]
    abs_path = f'/webhooks/{(path or func.__name__).strip('/')}'
    if abs_path in self.registered_webhooks:
        raise ValueError(f'Webhook {abs_path} already exists.')
    self.registered_webhooks[abs_path] = func