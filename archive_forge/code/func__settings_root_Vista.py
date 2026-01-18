import os
import platform
import pathlib
def _settings_root_Vista():
    return os.environ.get('LOCALAPPDATA', os.environ.get('ProgramData', '.'))