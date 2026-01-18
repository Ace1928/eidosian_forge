import os
import sys
from pathlib import Path
from textwrap import dedent
def _setupQtDirectories():
    pyside_package_dir = Path(__file__).parent.resolve()
    if sys.platform == 'win32' and sys.version_info[0] == 3 and (sys.version_info[1] >= 8):
        for dir in _additional_dll_directories(pyside_package_dir):
            os.add_dll_directory(os.fspath(dir))
    try:
        from shiboken6 import Shiboken
    except Exception:
        paths = ', '.join(sys.path)
        print(f'PySide6/__init__.py: Unable to import Shiboken from {paths}', file=sys.stderr)
        raise
    if sys.platform == 'win32':
        os.environ['PATH'] = os.fspath(pyside_package_dir) + os.pathsep + os.environ['PATH']
        openssl_dir = pyside_package_dir / 'openssl'
        if openssl_dir.exists():
            path = os.environ['PATH']
            try:
                os.environ['PATH'] = os.fspath(openssl_dir) + os.pathsep + path
                try:
                    from . import QtNetwork
                except ImportError:
                    pass
                else:
                    QtNetwork.QSslSocket.supportsSsl()
            finally:
                os.environ['PATH'] = path