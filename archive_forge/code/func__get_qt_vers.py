import os
import importlib
import pytest
from IPython.terminal.pt_inputhooks import set_qt_api, get_inputhook_name_and_func
def _get_qt_vers():
    """If any version of Qt is available, this will populate `guis_avail` with 'qt' and 'qtx'. Due
    to the import mechanism, we can't import multiple versions of Qt in one session."""
    for gui in ['qt', 'qt6', 'qt5']:
        print(f'Trying {gui}')
        try:
            set_qt_api(gui)
            importlib.import_module('IPython.terminal.pt_inputhooks.qt')
            guis_avail.append(gui)
            if 'QT_API' in os.environ.keys():
                del os.environ['QT_API']
        except ImportError:
            pass
        except RuntimeError:
            pass