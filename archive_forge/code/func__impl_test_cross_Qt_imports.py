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
def _impl_test_cross_Qt_imports():
    import sys
    import importlib
    import pytest
    _, host_binding, mpl_binding = sys.argv
    importlib.import_module(f'{mpl_binding}.QtCore')
    mpl_binding_qwidgets = importlib.import_module(f'{mpl_binding}.QtWidgets')
    import matplotlib.backends.backend_qt
    host_qwidgets = importlib.import_module(f'{host_binding}.QtWidgets')
    host_app = host_qwidgets.QApplication(['mpl testing'])
    with pytest.warns(UserWarning, match='Mixing Qt major'):
        matplotlib.backends.backend_qt._create_qApp()