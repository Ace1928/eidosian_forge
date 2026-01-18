import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from PIL import Image
import pytest
import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.testing import subprocess_run_helper as _run_helper
def _test_other_signal_before_sigint_impl(backend, target_name, kwargs):
    import signal
    import matplotlib.pyplot as plt
    plt.switch_backend(backend)
    target = getattr(plt, target_name)
    fig = plt.figure()
    fig.canvas.mpl_connect('draw_event', lambda *args: print('DRAW', flush=True))
    timer = fig.canvas.new_timer(interval=1)
    timer.single_shot = True
    timer.add_callback(print, 'SIGUSR1', flush=True)

    def custom_signal_handler(signum, frame):
        timer.start()
    signal.signal(signal.SIGUSR1, custom_signal_handler)
    try:
        target(**kwargs)
    except KeyboardInterrupt:
        print('SUCCESS', flush=True)