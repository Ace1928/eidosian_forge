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
def _lazy_headless():
    import os
    import sys
    backend, deps = sys.argv[1:]
    deps = deps.split(',')
    os.environ.pop('DISPLAY', None)
    os.environ.pop('WAYLAND_DISPLAY', None)
    for dep in deps:
        assert dep not in sys.modules
    import matplotlib.pyplot as plt
    assert plt.get_backend() == 'agg'
    for dep in deps:
        assert dep not in sys.modules
    for dep in deps:
        importlib.import_module(dep)
        assert dep in sys.modules
    try:
        plt.switch_backend(backend)
    except ImportError:
        pass
    else:
        sys.exit(1)