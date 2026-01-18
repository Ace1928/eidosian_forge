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
def _test_interactive_impl():
    import importlib.util
    import io
    import json
    import sys
    import pytest
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.backend_bases import KeyEvent
    mpl.rcParams.update({'webagg.open_in_browser': False, 'webagg.port_retries': 1})
    mpl.rcParams.update(json.loads(sys.argv[1]))
    backend = plt.rcParams['backend'].lower()
    if backend.endswith('agg') and (not backend.startswith(('gtk', 'web'))):
        plt.figure()
        if backend != 'tkagg':
            with pytest.raises(ImportError):
                mpl.use('tkagg', force=True)

        def check_alt_backend(alt_backend):
            mpl.use(alt_backend, force=True)
            fig = plt.figure()
            assert type(fig.canvas).__module__ == f'matplotlib.backends.backend_{alt_backend}'
            plt.close('all')
        if importlib.util.find_spec('cairocffi'):
            check_alt_backend(backend[:-3] + 'cairo')
        check_alt_backend('svg')
    mpl.use(backend, force=True)
    fig, ax = plt.subplots()
    assert type(fig.canvas).__module__ == f'matplotlib.backends.backend_{backend}'
    assert fig.canvas.manager.get_window_title() == 'Figure 1'
    if mpl.rcParams['toolbar'] == 'toolmanager':
        _test_toolbar_button_la_mode_icon(fig)
    ax.plot([0, 1], [2, 3])
    if fig.canvas.toolbar:
        fig.canvas.toolbar.draw_rubberband(None, 1.0, 1, 2.0, 2)
    timer = fig.canvas.new_timer(1.0)
    timer.add_callback(KeyEvent('key_press_event', fig.canvas, 'q')._process)
    fig.canvas.mpl_connect('draw_event', lambda event: timer.start())
    fig.canvas.mpl_connect('close_event', print)
    result = io.BytesIO()
    fig.savefig(result, format='png')
    plt.show()
    plt.pause(0.5)
    result_after = io.BytesIO()
    fig.savefig(result_after, format='png')
    if not backend.startswith('qt5') and sys.platform == 'darwin':
        assert result.getvalue() == result_after.getvalue()