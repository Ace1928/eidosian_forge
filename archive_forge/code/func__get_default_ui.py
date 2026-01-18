import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
def _get_default_ui(self) -> 'gr.Blocks':
    """Default UI if not provided (lists webhooks and provides basic instructions)."""
    import gradio as gr
    with gr.Blocks() as ui:
        gr.Markdown('# This is an app to process 🤗 Webhooks')
        gr.Markdown('Webhooks are a foundation for MLOps-related features. They allow you to listen for new changes on specific repos or to all repos belonging to particular set of users/organizations (not just your repos, but any repo). Check out this [guide](https://huggingface.co/docs/hub/webhooks) to get to know more about webhooks on the Huggingface Hub.')
        gr.Markdown(f'{len(self.registered_webhooks)} webhook(s) are registered:' + '\n\n' + '\n '.join((f'- [{webhook_path}]({_get_webhook_doc_url(webhook.__name__, webhook_path)})' for webhook_path, webhook in self.registered_webhooks.items())))
        gr.Markdown('Go to https://huggingface.co/settings/webhooks to setup your webhooks.' + '\nYou app is running locally. Please look at the logs to check the full URL you need to set.' if _is_local else "\nThis app is running on a Space. You can find the corresponding URL in the options menu (top-right) > 'Embed the Space'. The URL looks like 'https://{username}-{repo_name}.hf.space'.")
    return ui