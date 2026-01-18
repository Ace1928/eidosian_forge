import os
import sys
from pathlib import Path
from textwrap import dedent
def _additional_dll_directories(package_dir):
    root = Path(package_dir).parent
    if root.suffix == '.zip':
        return []
    shiboken6 = root / 'shiboken6'
    if shiboken6.is_dir():
        return [shiboken6]
    shiboken6 = Path(root).parent / 'shiboken6' / 'libshiboken'
    if not shiboken6.is_dir():
        raise ImportError(str(shiboken6) + ' does not exist')
    result = [shiboken6, root / 'libpyside']
    libpysideqml = root / 'libpysideqml'
    if libpysideqml.is_dir():
        result.append(libpysideqml)
    for path in os.environ.get('PATH').split(';'):
        if path:
            if (Path(path) / 'qmake.exe').exists():
                result.append(path)
                break
    return result