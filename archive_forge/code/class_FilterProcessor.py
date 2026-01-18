from .. import (
import stat
class FilterProcessor(processor.ImportProcessor):
    """An import processor that filters the input to include/exclude objects.

    No changes to the current repository are made.

    Here are the supported parameters:

    * include_paths - a list of paths that commits must change in order to
      be kept in the output stream

    * exclude_paths - a list of paths that should not appear in the output
      stream

    * squash_empty_commits - if set to False, squash commits that don't have
      any changes after the filter has been applied
    """
    known_params = [b'include_paths', b'exclude_paths', b'squash_empty_commits']

    def pre_process(self):
        self.includes = self.params.get(b'include_paths')
        self.excludes = self.params.get(b'exclude_paths')
        self.squash_empty_commits = bool(self.params.get(b'squash_empty_commits', True))
        self.new_root = helpers.common_directory(self.includes)
        self.blobs = {}
        self.squashed_commits = set()
        self.parents = {}

    def pre_handler(self, cmd):
        self.command = cmd
        self.keep = False
        self.referenced_blobs = []

    def post_handler(self, cmd):
        if not self.keep:
            return
        for blob_id in self.referenced_blobs:
            self._print_command(self.blobs[blob_id])
        self._print_command(self.command)

    def progress_handler(self, cmd):
        """Process a ProgressCommand."""
        self.keep = True

    def blob_handler(self, cmd):
        """Process a BlobCommand."""
        self.blobs[cmd.id] = cmd
        self.keep = False

    def checkpoint_handler(self, cmd):
        """Process a CheckpointCommand."""
        self.keep = True

    def commit_handler(self, cmd):
        """Process a CommitCommand."""
        interesting_filecmds = self._filter_filecommands(cmd.iter_files)
        if interesting_filecmds or not self.squash_empty_commits:
            if len(interesting_filecmds) == 1 and isinstance(interesting_filecmds[0], commands.FileDeleteAllCommand):
                pass
            else:
                self.keep = True
                cmd.file_iter = iter(interesting_filecmds)
                for fc in interesting_filecmds:
                    if isinstance(fc, commands.FileModifyCommand):
                        if fc.dataref is not None and (not stat.S_ISDIR(fc.mode)):
                            self.referenced_blobs.append(fc.dataref)
                cmd.from_ = self._find_interesting_from(cmd.from_)
                cmd.merges = self._find_interesting_merges(cmd.merges)
        else:
            self.squashed_commits.add(cmd.id)
        if cmd.from_ and cmd.merges:
            parents = [cmd.from_] + cmd.merges
        elif cmd.from_:
            parents = [cmd.from_]
        else:
            parents = None
        if cmd.mark is not None:
            self.parents[b':' + cmd.mark] = parents

    def reset_handler(self, cmd):
        """Process a ResetCommand."""
        if cmd.from_ is None:
            self.keep = True
        else:
            cmd.from_ = self._find_interesting_from(cmd.from_)
            self.keep = cmd.from_ is not None

    def tag_handler(self, cmd):
        """Process a TagCommand."""
        cmd.from_ = self._find_interesting_from(cmd.from_)
        self.keep = cmd.from_ is not None

    def feature_handler(self, cmd):
        """Process a FeatureCommand."""
        feature = cmd.feature_name
        if feature not in commands.FEATURE_NAMES:
            self.warning('feature %s is not supported - parsing may fail' % (feature,))
        self.keep = True

    def _print_command(self, cmd):
        """Wrapper to avoid adding unnecessary blank lines."""
        text = bytes(cmd)
        self.outf.write(text)
        if not text.endswith(b'\n'):
            self.outf.write(b'\n')

    def _filter_filecommands(self, filecmd_iter):
        """Return the filecommands filtered by includes & excludes.

        :return: a list of FileCommand objects
        """
        if self.includes is None and self.excludes is None:
            return list(filecmd_iter())
        result = []
        for fc in filecmd_iter():
            if isinstance(fc, commands.FileModifyCommand) or isinstance(fc, commands.FileDeleteCommand):
                if self._path_to_be_kept(fc.path):
                    fc.path = self._adjust_for_new_root(fc.path)
                else:
                    continue
            elif isinstance(fc, commands.FileDeleteAllCommand):
                pass
            elif isinstance(fc, commands.FileRenameCommand):
                fc = self._convert_rename(fc)
            elif isinstance(fc, commands.FileCopyCommand):
                fc = self._convert_copy(fc)
            else:
                self.warning('cannot handle FileCommands of class %s - ignoring', fc.__class__)
                continue
            if fc is not None:
                result.append(fc)
        return result

    def _path_to_be_kept(self, path):
        """Does the given path pass the filtering criteria?"""
        if self.excludes and (path in self.excludes or helpers.is_inside_any(self.excludes, path)):
            return False
        if self.includes:
            return path in self.includes or helpers.is_inside_any(self.includes, path)
        return True

    def _adjust_for_new_root(self, path):
        """Adjust a path given the new root directory of the output."""
        if self.new_root is None:
            return path
        elif path.startswith(self.new_root):
            return path[len(self.new_root):]
        else:
            return path

    def _find_interesting_parent(self, commit_ref):
        while True:
            if commit_ref not in self.squashed_commits:
                return commit_ref
            parents = self.parents.get(commit_ref)
            if not parents:
                return None
            commit_ref = parents[0]

    def _find_interesting_from(self, commit_ref):
        if commit_ref is None:
            return None
        return self._find_interesting_parent(commit_ref)

    def _find_interesting_merges(self, commit_refs):
        if commit_refs is None:
            return None
        merges = []
        for commit_ref in commit_refs:
            parent = self._find_interesting_parent(commit_ref)
            if parent is not None:
                merges.append(parent)
        if merges:
            return merges
        else:
            return None

    def _convert_rename(self, fc):
        """Convert a FileRenameCommand into a new FileCommand.

        :return: None if the rename is being ignored, otherwise a
          new FileCommand based on the whether the old and new paths
          are inside or outside of the interesting locations.
          """
        old = fc.old_path
        new = fc.new_path
        keep_old = self._path_to_be_kept(old)
        keep_new = self._path_to_be_kept(new)
        if keep_old and keep_new:
            fc.old_path = self._adjust_for_new_root(old)
            fc.new_path = self._adjust_for_new_root(new)
            return fc
        elif keep_old:
            old = self._adjust_for_new_root(old)
            return commands.FileDeleteCommand(old)
        elif keep_new:
            self.warning('cannot turn rename of %s into an add of %s yet' % (old, new))
        return None

    def _convert_copy(self, fc):
        """Convert a FileCopyCommand into a new FileCommand.

        :return: None if the copy is being ignored, otherwise a
          new FileCommand based on the whether the source and destination
          paths are inside or outside of the interesting locations.
          """
        src = fc.src_path
        dest = fc.dest_path
        keep_src = self._path_to_be_kept(src)
        keep_dest = self._path_to_be_kept(dest)
        if keep_src and keep_dest:
            fc.src_path = self._adjust_for_new_root(src)
            fc.dest_path = self._adjust_for_new_root(dest)
            return fc
        elif keep_src:
            return None
        elif keep_dest:
            self.warning('cannot turn copy of %s into an add of %s yet' % (src, dest))
        return None