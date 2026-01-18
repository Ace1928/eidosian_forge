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
def _impl_missing():
    import sys
    sys.modules['PyQt6'] = None
    sys.modules['PyQt5'] = None
    sys.modules['PySide2'] = None
    sys.modules['PySide6'] = None
    import matplotlib.pyplot as plt
    with pytest.raises(ImportError, match='Failed to import any of the following Qt'):
        plt.switch_backend('qtagg')
    with pytest.raises(ImportError, match='^(?:(?!(PySide6|PyQt6)).)*$'):
        plt.switch_backend('qt5agg')