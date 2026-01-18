import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
def _warn_on_empty_secret(webhook_secret: Optional[str]) -> None:
    if webhook_secret is None:
        print('Webhook secret is not defined. This means your webhook endpoints will be open to everyone.')
        print("To add a secret, set `WEBHOOK_SECRET` as environment variable or pass it at initialization: \n\t`app = WebhooksServer(webhook_secret='my_secret', ...)`")
        print('For more details about webhook secrets, please refer to https://huggingface.co/docs/hub/webhooks#webhook-secret.')
    else:
        print('Webhook secret is correctly defined.')