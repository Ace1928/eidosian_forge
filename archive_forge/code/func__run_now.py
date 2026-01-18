import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
@wraps(app.run)
def _run_now():
    atexit.unregister(app.run)
    app.run()