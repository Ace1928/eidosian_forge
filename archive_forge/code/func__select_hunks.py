import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def _select_hunks(self, creator, file_id, work_tree_lines):
    """Provide diff hunk selection for modified text.

        If self.reporter.invert_diff is True, the diff is inverted so that
        insertions are displayed as removals and vice versa.

        :param creator: a ShelfCreator
        :param file_id: The id of the file to shelve.
        :param work_tree_lines: Line contents of the file in the working tree.
        :return: number of shelved hunks.
        """
    if self.reporter.invert_diff:
        target_lines = work_tree_lines
    else:
        path = self.target_tree.id2path(file_id)
        target_lines = self.target_tree.get_file_lines(path)
    textfile.check_text_lines(work_tree_lines)
    textfile.check_text_lines(target_lines)
    parsed = self.get_parsed_patch(file_id, self.reporter.invert_diff)
    final_hunks = []
    if not self.auto:
        offset = 0
        self.diff_writer.write(parsed.get_header())
        for hunk in parsed.hunks:
            self.diff_writer.write(hunk.as_bytes())
            selected = self.prompt_bool(self.reporter.vocab['hunk'], allow_editor=self.change_editor is not None)
            if not self.reporter.invert_diff:
                selected = not selected
            if selected:
                hunk.mod_pos += offset
                final_hunks.append(hunk)
            else:
                offset -= hunk.mod_range - hunk.orig_range
    sys.stdout.flush()
    if self.reporter.invert_diff:
        change_count = len(final_hunks)
    else:
        change_count = len(parsed.hunks) - len(final_hunks)
    patched = patches.iter_patched_from_hunks(target_lines, final_hunks)
    lines = list(patched)
    return (lines, change_count)