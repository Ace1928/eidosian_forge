import os
import warnings
from typing import Optional, Sequence, Mapping, Callable
from ipywidgets import Dropdown, Text, Select, Button, HTML
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget
from .errors import ParentPathError, InvalidFileNameError
from .utils import get_subpaths, get_dir_contents, match_item, strip_parent_path
from .utils import is_valid_filename, get_drive_letters, normalize_path, has_parent_path
def _on_dircontent_select(self, change: Mapping[str, str]) -> None:
    """Handle selecting a folder entry."""
    new_path = os.path.realpath(os.path.join(self._expand_path(self._pathlist.value), self._map_disp_to_name[change['new']]))
    if os.path.isdir(new_path):
        path = new_path
        filename = self._filename.value
    else:
        path = self._expand_path(self._pathlist.value)
        filename = self._map_disp_to_name[change['new']]
    self._set_form_values(path, filename)