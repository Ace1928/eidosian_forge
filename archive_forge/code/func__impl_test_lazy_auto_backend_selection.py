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
def _impl_test_lazy_auto_backend_selection():
    import matplotlib
    import matplotlib.pyplot as plt
    bk = matplotlib.rcParams._get('backend')
    assert not isinstance(bk, str)
    assert plt._backend_mod is None
    plt.plot(5)
    assert plt._backend_mod is not None
    bk = matplotlib.rcParams._get('backend')
    assert isinstance(bk, str)