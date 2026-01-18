import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
def _get_webhook_doc_url(webhook_name: str, webhook_path: str) -> str:
    """Returns the anchor to a given webhook in the docs (experimental)"""
    return '/docs#/default/' + webhook_name + webhook_path.replace('/', '_') + '_post'