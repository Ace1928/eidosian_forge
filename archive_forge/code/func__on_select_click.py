import os
import warnings
from typing import Optional, Sequence, Mapping, Callable
from ipywidgets import Dropdown, Text, Select, Button, HTML
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget
from .errors import ParentPathError, InvalidFileNameError
from .utils import get_subpaths, get_dir_contents, match_item, strip_parent_path
from .utils import is_valid_filename, get_drive_letters, normalize_path, has_parent_path
def _on_select_click(self, _b) -> None:
    """Handle select button clicks."""
    if self._gb.layout.display == 'none':
        self._show_dialog()
    else:
        self._apply_selection()
        if self._callback is not None:
            try:
                self._callback(self)
            except TypeError:
                self._callback()