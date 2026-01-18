import stat
from typing import Dict, Tuple
from fastimport import commands, parser, processor
from fastimport import errors as fastimport_errors
from .index import commit_tree
from .object_store import iter_tree_contents
from .objects import ZERO_SHA, Blob, Commit, Tag
class GitFastExporter:
    """Generate a fast-export output stream for Git objects."""

    def __init__(self, outf, store) -> None:
        self.outf = outf
        self.store = store
        self.markers: Dict[bytes, bytes] = {}
        self._marker_idx = 0

    def print_cmd(self, cmd):
        self.outf.write(getattr(cmd, '__bytes__', cmd.__repr__)() + b'\n')

    def _allocate_marker(self):
        self._marker_idx += 1
        return ('%d' % (self._marker_idx,)).encode('ascii')

    def _export_blob(self, blob):
        marker = self._allocate_marker()
        self.markers[marker] = blob.id
        return (commands.BlobCommand(marker, blob.data), marker)

    def emit_blob(self, blob):
        cmd, marker = self._export_blob(blob)
        self.print_cmd(cmd)
        return marker

    def _iter_files(self, base_tree, new_tree):
        for (old_path, new_path), (old_mode, new_mode), (old_hexsha, new_hexsha) in self.store.tree_changes(base_tree, new_tree):
            if new_path is None:
                yield commands.FileDeleteCommand(old_path)
                continue
            if not stat.S_ISDIR(new_mode):
                blob = self.store[new_hexsha]
                marker = self.emit_blob(blob)
            if old_path != new_path and old_path is not None:
                yield commands.FileRenameCommand(old_path, new_path)
            if old_mode != new_mode or old_hexsha != new_hexsha:
                prefixed_marker = b':' + marker
                yield commands.FileModifyCommand(new_path, new_mode, prefixed_marker, None)

    def _export_commit(self, commit, ref, base_tree=None):
        file_cmds = list(self._iter_files(base_tree, commit.tree))
        marker = self._allocate_marker()
        if commit.parents:
            from_ = commit.parents[0]
            merges = commit.parents[1:]
        else:
            from_ = None
            merges = []
        author, author_email = split_email(commit.author)
        committer, committer_email = split_email(commit.committer)
        cmd = commands.CommitCommand(ref, marker, (author, author_email, commit.author_time, commit.author_timezone), (committer, committer_email, commit.commit_time, commit.commit_timezone), commit.message, from_, merges, file_cmds)
        return (cmd, marker)

    def emit_commit(self, commit, ref, base_tree=None):
        cmd, marker = self._export_commit(commit, ref, base_tree)
        self.print_cmd(cmd)
        return marker