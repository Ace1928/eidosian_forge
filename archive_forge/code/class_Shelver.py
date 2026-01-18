import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
class Shelver:
    """Interactively shelve the changes in a working tree."""

    def __init__(self, work_tree, target_tree, diff_writer=None, auto=False, auto_apply=False, file_list=None, message=None, destroy=False, manager=None, reporter=None):
        """Constructor.

        :param work_tree: The working tree to shelve changes from.
        :param target_tree: The "unchanged" / old tree to compare the
            work_tree to.
        :param auto: If True, shelve each possible change.
        :param auto_apply: If True, shelve changes with no final prompt.
        :param file_list: If supplied, only files in this list may be shelved.
        :param message: The message to associate with the shelved changes.
        :param destroy: Change the working tree without storing the shelved
            changes.
        :param manager: The shelf manager to use.
        :param reporter: Object for reporting changes to user.
        """
        self.work_tree = work_tree
        self.target_tree = target_tree
        self.diff_writer = diff_writer
        if self.diff_writer is None:
            self.diff_writer = sys.stdout
        if manager is None:
            manager = work_tree.get_shelf_manager()
        self.manager = manager
        self.auto = auto
        self.auto_apply = auto_apply
        self.file_list = file_list
        self.message = message
        self.destroy = destroy
        if reporter is None:
            reporter = ShelfReporter()
        self.reporter = reporter
        config = self.work_tree.branch.get_config()
        self.change_editor = config.get_change_editor(target_tree, work_tree)
        self.work_tree.lock_tree_write()

    @classmethod
    def from_args(klass, diff_writer, revision=None, all=False, file_list=None, message=None, directory=None, destroy=False):
        """Create a shelver from commandline arguments.

        The returned shelver wil have a work_tree that is locked and should
        be unlocked.

        :param revision: RevisionSpec of the revision to compare to.
        :param all: If True, shelve all changes without prompting.
        :param file_list: If supplied, only files in this list may be  shelved.
        :param message: The message to associate with the shelved changes.
        :param directory: The directory containing the working tree.
        :param destroy: Change the working tree without storing the shelved
            changes.
        """
        if directory is None:
            directory = '.'
        elif file_list:
            file_list = [osutils.pathjoin(directory, f) for f in file_list]
        tree, path = workingtree.WorkingTree.open_containing(directory)
        with tree.lock_tree_write():
            target_tree = builtins._get_one_revision_tree('shelf2', revision, tree.branch, tree)
            files = tree.safe_relpath_files(file_list)
            return klass(tree, target_tree, diff_writer, all, all, files, message, destroy)

    def run(self):
        """Interactively shelve the changes."""
        creator = shelf.ShelfCreator(self.work_tree, self.target_tree, self.file_list)
        self.tempdir = tempfile.mkdtemp()
        changes_shelved = 0
        try:
            for change in creator.iter_shelvable():
                if change[0] == 'modify text':
                    try:
                        changes_shelved += self.handle_modify_text(creator, change[1])
                    except errors.BinaryFile:
                        if self.prompt_bool(self.reporter.vocab['binary']):
                            changes_shelved += 1
                            creator.shelve_content_change(change[1])
                elif self.prompt_bool(self.reporter.prompt_change(change)):
                    creator.shelve_change(change)
                    changes_shelved += 1
            if changes_shelved > 0:
                self.reporter.selected_changes(creator.work_transform)
                if self.auto_apply or self.prompt_bool(self.reporter.vocab['final'] % changes_shelved):
                    if self.destroy:
                        creator.transform()
                        self.reporter.changes_destroyed()
                    else:
                        shelf_id = self.manager.shelve_changes(creator, self.message)
                        self.reporter.shelved_id(shelf_id)
            else:
                self.reporter.no_changes()
        finally:
            shutil.rmtree(self.tempdir)
            creator.finalize()

    def finalize(self):
        if self.change_editor is not None:
            self.change_editor.finish()
        self.work_tree.unlock()

    def get_parsed_patch(self, file_id, invert=False):
        """Return a parsed version of a file's patch.

        :param file_id: The id of the file to generate a patch for.
        :param invert: If True, provide an inverted patch (insertions displayed
            as removals, removals displayed as insertions).
        :return: A patches.Patch.
        """
        diff_file = BytesIO()
        if invert:
            old_tree = self.work_tree
            new_tree = self.target_tree
        else:
            old_tree = self.target_tree
            new_tree = self.work_tree
        old_path = old_tree.id2path(file_id)
        new_path = new_tree.id2path(file_id)
        path_encoding = osutils.get_terminal_encoding()
        text_differ = diff.DiffText(old_tree, new_tree, diff_file, path_encoding=path_encoding)
        patch = text_differ.diff(old_path, new_path, 'file', 'file')
        diff_file.seek(0)
        return patches.parse_patch(diff_file)

    def prompt(self, message, choices, default):
        return ui.ui_factory.choose(message, choices, default=default)

    def prompt_bool(self, question, allow_editor=False):
        """Prompt the user with a yes/no question.

        This may be overridden by self.auto.  It may also *set* self.auto.  It
        may also raise UserAbort.
        :param question: The question to ask the user.
        :return: True or False
        """
        if self.auto:
            return True
        alternatives_chars = 'yn'
        alternatives = '&yes\n&No'
        if allow_editor:
            alternatives_chars += 'e'
            alternatives += '\n&edit manually'
        alternatives_chars += 'fq'
        alternatives += '\n&finish\n&quit'
        choice = self.prompt(question, alternatives, 1)
        if choice is None:
            char = 'n'
        else:
            char = alternatives_chars[choice]
        if char == 'y':
            return True
        elif char == 'e' and allow_editor:
            raise UseEditor
        elif char == 'f':
            self.auto = True
            return True
        if char == 'q':
            raise errors.UserAbort()
        else:
            return False

    def handle_modify_text(self, creator, file_id):
        """Handle modified text, by using hunk selection or file editing.

        :param creator: A ShelfCreator.
        :param file_id: The id of the file that was modified.
        :return: The number of changes.
        """
        path = self.work_tree.id2path(file_id)
        work_tree_lines = self.work_tree.get_file_lines(path, file_id)
        try:
            lines, change_count = self._select_hunks(creator, file_id, work_tree_lines)
        except UseEditor:
            lines, change_count = self._edit_file(file_id, work_tree_lines)
        if change_count != 0:
            creator.shelve_lines(file_id, lines)
        return change_count

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

    @staticmethod
    def _count_changed_regions(old_lines, new_lines):
        matcher = patiencediff.PatienceSequenceMatcher(None, old_lines, new_lines)
        blocks = matcher.get_matching_blocks()
        return len(blocks) - 2