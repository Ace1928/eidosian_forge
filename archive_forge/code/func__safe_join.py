from __future__ import annotations
import fnmatch
import os
import warnings
from pathlib import Path
from typing import Any, Callable, List, Literal
from gradio_client.documentation import document
from gradio.components.base import Component, server
from gradio.data_classes import GradioRootModel
def _safe_join(self, folders):
    combined_path = os.path.join(self.root_dir, *folders)
    absolute_path = os.path.abspath(combined_path)
    if os.path.commonprefix([self.root_dir, absolute_path]) != os.path.abspath(self.root_dir):
        raise ValueError('Attempted to navigate outside of root directory')
    return absolute_path