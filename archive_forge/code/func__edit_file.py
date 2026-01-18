import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def _edit_file(self, file_id, work_tree_lines):
    """
        :param file_id: id of the file to edit.
        :param work_tree_lines: Line contents of the file in the working tree.
        :return: (lines, change_region_count), where lines is the new line
            content of the file, and change_region_count is the number of
            changed regions.
        """
    lines = osutils.split_lines(self.change_editor.edit_file(self.change_editor.old_tree.id2path(file_id), self.change_editor.new_tree.id2path(file_id)))
    return (lines, self._count_changed_regions(work_tree_lines, lines))