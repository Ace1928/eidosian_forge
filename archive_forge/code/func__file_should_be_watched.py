from __future__ import annotations
import collections
import os
import sys
import types
from pathlib import Path
from typing import Callable, Final
from streamlit import config, file_util
from streamlit.folder_black_list import FolderBlackList
from streamlit.logger import get_logger
from streamlit.source_util import get_pages
from streamlit.watcher.path_watcher import (
def _file_should_be_watched(self, filepath):
    return self._file_is_new(filepath) and (file_util.file_is_in_folder_glob(filepath, self._script_folder) or file_util.file_in_pythonpath(filepath))