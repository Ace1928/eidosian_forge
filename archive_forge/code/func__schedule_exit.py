import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def _schedule_exit(delay):
    """schedule fall back to main loop in [delay] seconds"""
    app.after(int(1000 * delay), exit_loop)