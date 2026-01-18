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
def _get_testable_interactive_backends():
    envs = []
    for deps, env in [*[([qt_api], {'MPLBACKEND': 'qtagg', 'QT_API': qt_api}) for qt_api in ['PyQt6', 'PySide6', 'PyQt5', 'PySide2']], *[([qt_api, 'cairocffi'], {'MPLBACKEND': 'qtcairo', 'QT_API': qt_api}) for qt_api in ['PyQt6', 'PySide6', 'PyQt5', 'PySide2']], *[(['cairo', 'gi'], {'MPLBACKEND': f'gtk{version}{renderer}'}) for version in [3, 4] for renderer in ['agg', 'cairo']], (['tkinter'], {'MPLBACKEND': 'tkagg'}), (['wx'], {'MPLBACKEND': 'wx'}), (['wx'], {'MPLBACKEND': 'wxagg'}), (['matplotlib.backends._macosx'], {'MPLBACKEND': 'macosx'})]:
        reason = None
        missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
        if sys.platform == 'linux' and (not _c_internal_utils.display_is_valid()):
            reason = '$DISPLAY and $WAYLAND_DISPLAY are unset'
        elif missing:
            reason = '{} cannot be imported'.format(', '.join(missing))
        elif env['MPLBACKEND'] == 'macosx' and os.environ.get('TF_BUILD'):
            reason = 'macosx backend fails on Azure'
        elif env['MPLBACKEND'].startswith('gtk'):
            import gi
            version = env['MPLBACKEND'][3]
            repo = gi.Repository.get_default()
            if f'{version}.0' not in repo.enumerate_versions('Gtk'):
                reason = 'no usable GTK bindings'
        marks = []
        if reason:
            marks.append(pytest.mark.skip(reason=f'Skipping {env} because {reason}'))
        elif env['MPLBACKEND'].startswith('wx') and sys.platform == 'darwin':
            marks.append(pytest.mark.xfail(reason='github #16849'))
        elif env['MPLBACKEND'] == 'tkagg' and ('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and (sys.platform == 'darwin') and (sys.version_info[:2] < (3, 11)):
            marks.append(pytest.mark.xfail(reason='Tk version mismatch on Azure macOS CI'))
        envs.append(pytest.param({**env, 'BACKEND_DEPS': ','.join(deps)}, marks=marks, id=str(env)))
    return envs