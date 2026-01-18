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
def _display_in_colab(dashboard_url, port, mode, width, height):
    from google.colab import output
    if mode == 'inline':
        output.serve_kernel_port_as_iframe(port, width=width, height=height)
    elif mode == 'external':
        print('Dash app running on:')
        output.serve_kernel_port_as_window(port, anchor_text=dashboard_url)