import os
import warnings
from typing import Optional, Sequence, Mapping, Callable
from ipywidgets import Dropdown, Text, Select, Button, HTML
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget
from .errors import ParentPathError, InvalidFileNameError
from .utils import get_subpaths, get_dir_contents, match_item, strip_parent_path
from .utils import is_valid_filename, get_drive_letters, normalize_path, has_parent_path
def _show_dialog(self) -> None:
    """Show the dialog."""
    self._gb.layout.display = None
    self._cancel.layout.display = None
    if self._selected_path is not None and self._selected_filename is not None:
        path = self._selected_path
        filename = self._selected_filename
    else:
        path = self._default_path
        filename = self._default_filename
    self._set_form_values(path, filename)