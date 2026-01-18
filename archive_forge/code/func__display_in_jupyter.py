import asyncio
import io
import inspect
import logging
import os
import queue
import uuid
import sys
import threading
import time
from typing_extensions import Literal
from werkzeug.serving import make_server
@staticmethod
def _display_in_jupyter(dashboard_url, port, mode, width, height):
    if mode == 'inline':
        display(IFrame(dashboard_url, width, height))
    elif mode in ('external', 'tab'):
        print(f'Dash app running on {dashboard_url}')
        if mode == 'tab':
            display(Javascript(f"window.open('{dashboard_url}')"))
    elif mode == 'jupyterlab':
        _dash_comm.send({'type': 'show', 'port': port, 'url': dashboard_url})