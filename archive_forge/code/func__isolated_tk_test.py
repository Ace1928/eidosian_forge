import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper
def _isolated_tk_test(success_count, func=None):
    """
    A decorator to run *func* in a subprocess and assert that it prints
    "success" *success_count* times and nothing on stderr.

    TkAgg tests seem to have interactions between tests, so isolate each test
    in a subprocess. See GH#18261
    """
    if func is None:
        return functools.partial(_isolated_tk_test, success_count)
    if 'MPL_TEST_ESCAPE_HATCH' in os.environ:
        return func

    @pytest.mark.skipif(not importlib.util.find_spec('tkinter'), reason='missing tkinter')
    @pytest.mark.skipif(sys.platform == 'linux' and (not _c_internal_utils.display_is_valid()), reason='$DISPLAY and $WAYLAND_DISPLAY are unset')
    @pytest.mark.xfail(('TF_BUILD' in os.environ or 'GITHUB_ACTION' in os.environ) and sys.platform == 'darwin' and (sys.version_info[:2] < (3, 11)), reason='Tk version mismatch on Azure macOS CI')
    @functools.wraps(func)
    def test_func():
        pytest.importorskip('tkinter')
        try:
            proc = subprocess_run_helper(func, timeout=_test_timeout, extra_env=dict(MPLBACKEND='TkAgg', MPL_TEST_ESCAPE_HATCH='1'))
        except subprocess.TimeoutExpired:
            pytest.fail('Subprocess timed out')
        except subprocess.CalledProcessError as e:
            pytest.fail('Subprocess failed to test intended behavior\n' + str(e.stderr))
        else:
            ignored_lines = ['OpenGL', 'CFMessagePort: bootstrap_register', '/usr/include/servers/bootstrap_defs.h']
            assert not [line for line in proc.stderr.splitlines() if all((msg not in line for msg in ignored_lines))]
            assert proc.stdout.count('success') == success_count
    return test_func