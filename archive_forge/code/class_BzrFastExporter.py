import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
class BzrFastExporter:

    def __init__(self, source, outf, ref=None, checkpoint=-1, import_marks_file=None, export_marks_file=None, revision=None, verbose=False, plain_format=False, rewrite_tags=False, no_tags=False, baseline=False):
        """Export branch data in fast import format.

        :param plain_format: if True, 'classic' fast-import format is
            used without any extended features; if False, the generated
            data is richer and includes information like multiple
            authors, revision properties, etc.
        :param rewrite_tags: if True and if plain_format is set, tag names
            will be rewritten to be git-compatible.
            Otherwise tags which aren't valid for git will be skipped if
            plain_format is set.
        :param no_tags: if True tags won't be exported at all
        """
        self.branch = source
        self.outf = outf
        self.ref = ref
        self.checkpoint = checkpoint
        self.import_marks_file = import_marks_file
        self.export_marks_file = export_marks_file
        self.revision = revision
        self.excluded_revisions = set()
        self.plain_format = plain_format
        self.rewrite_tags = rewrite_tags
        self.no_tags = no_tags
        self.baseline = baseline
        self.tree_cache = lru_cache.LRUCache(max_cache=20)
        self._multi_author_api_available = hasattr(breezy.revision.Revision, 'get_apparent_authors')
        self.properties_to_exclude = ['authors', 'author']
        self.verbose = verbose
        if verbose:
            self.progress_every = 100
        else:
            self.progress_every = 1000
        self._start_time = time.time()
        self._commit_total = 0
        self.revid_to_mark = {}
        self.branch_names = {}
        if self.import_marks_file:
            marks_info = marks_file.import_marks(self.import_marks_file)
            if marks_info is not None:
                self.revid_to_mark = {r: m for m, r in marks_info.items()}

    def interesting_history(self):
        if self.revision:
            rev1, rev2 = builtins._get_revision_range(self.revision, self.branch, 'fast-export')
            start_rev_id = rev1.rev_id
            end_rev_id = rev2.rev_id
        else:
            start_rev_id = None
            end_rev_id = None
        self.note('Calculating the revisions to include ...')
        view_revisions = [rev_id for rev_id, _, _, _ in self.branch.iter_merge_sorted_revisions(end_rev_id, start_rev_id)]
        view_revisions.reverse()
        if start_rev_id is not None:
            self.note('Calculating the revisions to exclude ...')
            self.excluded_revisions = {rev_id for rev_id, _, _, _ in self.branch.iter_merge_sorted_revisions(start_rev_id)}
            if self.baseline:
                self.excluded_revisions.remove(start_rev_id)
                view_revisions.insert(0, start_rev_id)
        return list(view_revisions)

    def emit_commits(self, interesting):
        if self.baseline:
            revobj = self.branch.repository.get_revision(interesting.pop(0))
            self.emit_baseline(revobj, self.ref)
        for i in range(0, len(interesting), REVISIONS_CHUNK_SIZE):
            chunk = interesting[i:i + REVISIONS_CHUNK_SIZE]
            history = dict(self.branch.repository.iter_revisions(chunk))
            trees_needed = set()
            trees = {}
            for revid in chunk:
                trees_needed.update(self.preprocess_commit(revid, history[revid], self.ref))
            for tree in self._get_revision_trees(trees_needed):
                trees[tree.get_revision_id()] = tree
            for revid in chunk:
                revobj = history[revid]
                if len(revobj.parent_ids) == 0:
                    parent = breezy.revision.NULL_REVISION
                else:
                    parent = revobj.parent_ids[0]
                self.emit_commit(revobj, self.ref, trees[parent], trees[revid])

    def run(self):
        with self.branch.repository.lock_read():
            interesting = self.interesting_history()
            self._commit_total = len(interesting)
            self.note('Starting export of %d revisions ...' % self._commit_total)
            if not self.plain_format:
                self.emit_features()
            self.emit_commits(interesting)
            if self.branch.supports_tags() and (not self.no_tags):
                self.emit_tags()
        self._save_marks()
        self.dump_stats()

    def note(self, msg, *args):
        """Output a note but timestamp it."""
        msg = '{} {}'.format(self._time_of_day(), msg)
        trace.note(msg, *args)

    def warning(self, msg, *args):
        """Output a warning but timestamp it."""
        msg = '{} WARNING: {}'.format(self._time_of_day(), msg)
        trace.warning(msg, *args)

    def _time_of_day(self):
        """Time of day as a string."""
        return time.strftime('%H:%M:%S')

    def report_progress(self, commit_count, details=''):
        if commit_count and commit_count % self.progress_every == 0:
            if self._commit_total:
                counts = '%d/%d' % (commit_count, self._commit_total)
            else:
                counts = '%d' % (commit_count,)
            minutes = (time.time() - self._start_time) / 60
            rate = commit_count * 1.0 / minutes
            if rate > 10:
                rate_str = 'at %.0f/minute ' % rate
            else:
                rate_str = 'at %.1f/minute ' % rate
            self.note('{} commits exported {}{}'.format(counts, rate_str, details))

    def dump_stats(self):
        time_required = progress.str_tdelta(time.time() - self._start_time)
        rc = len(self.revid_to_mark)
        self.note('Exported %d %s in %s', rc, helpers.single_plural(rc, 'revision', 'revisions'), time_required)

    def print_cmd(self, cmd):
        self.outf.write(b'%s\n' % cmd)

    def _save_marks(self):
        if self.export_marks_file:
            revision_ids = {m: r for r, m in self.revid_to_mark.items()}
            marks_file.export_marks(self.export_marks_file, revision_ids)

    def is_empty_dir(self, tree, path):
        try:
            if tree.kind(path) != 'directory':
                return False
        except _mod_transport.NoSuchFile:
            self.warning('Skipping empty_dir detection - no file_id for %s' % (path,))
            return False
        contents = list(tree.walkdirs(prefix=path))[0]
        if len(contents[1]) == 0:
            return True
        else:
            return False

    def emit_features(self):
        for feature in sorted(commands.FEATURE_NAMES):
            self.print_cmd(commands.FeatureCommand(feature))

    def emit_baseline(self, revobj, ref):
        mark = 1
        self.revid_to_mark[revobj.revision_id] = b'%d' % mark
        tree_old = self.branch.repository.revision_tree(breezy.revision.NULL_REVISION)
        [tree_new] = list(self._get_revision_trees([revobj.revision_id]))
        file_cmds = self._get_filecommands(tree_old, tree_new)
        self.print_cmd(commands.ResetCommand(ref, None))
        self.print_cmd(self._get_commit_command(ref, mark, revobj, file_cmds))

    def preprocess_commit(self, revid, revobj, ref):
        if self.revid_to_mark.get(revid) or revid in self.excluded_revisions:
            return []
        if revobj is None:
            self.revid_to_mark[revid] = None
            return []
        if len(revobj.parent_ids) == 0:
            parent = breezy.revision.NULL_REVISION
        else:
            parent = revobj.parent_ids[0]
        self.revid_to_mark[revobj.revision_id] = b'%d' % (len(self.revid_to_mark) + 1)
        return [parent, revobj.revision_id]

    def emit_commit(self, revobj, ref, tree_old, tree_new):
        if tree_old.get_revision_id() == breezy.revision.NULL_REVISION:
            self.print_cmd(commands.ResetCommand(ref, None))
        file_cmds = self._get_filecommands(tree_old, tree_new)
        mark = self.revid_to_mark[revobj.revision_id]
        self.print_cmd(self._get_commit_command(ref, mark, revobj, file_cmds))
        ncommits = len(self.revid_to_mark)
        self.report_progress(ncommits)
        if self.checkpoint is not None and self.checkpoint > 0 and ncommits and (ncommits % self.checkpoint == 0):
            self.note('Exported %i commits - adding checkpoint to output' % ncommits)
            self._save_marks()
            self.print_cmd(commands.CheckpointCommand())

    def _get_name_email(self, user):
        if user.find('<') == -1:
            name = user
            email = ''
        else:
            name, email = parseaddr(user)
        return (name.encode('utf-8'), email.encode('utf-8'))

    def _get_commit_command(self, git_ref, mark, revobj, file_cmds):
        committer = revobj.committer
        name, email = self._get_name_email(committer)
        committer_info = (name, email, revobj.timestamp, revobj.timezone)
        if self._multi_author_api_available:
            more_authors = revobj.get_apparent_authors()
            author = more_authors.pop(0)
        else:
            more_authors = []
            author = revobj.get_apparent_author()
        if not self.plain_format and more_authors:
            name, email = self._get_name_email(author)
            author_info = (name, email, revobj.timestamp, revobj.timezone)
            more_author_info = []
            for a in more_authors:
                name, email = self._get_name_email(a)
                more_author_info.append((name, email, revobj.timestamp, revobj.timezone))
        elif author != committer:
            name, email = self._get_name_email(author)
            author_info = (name, email, revobj.timestamp, revobj.timezone)
            more_author_info = None
        else:
            author_info = None
            more_author_info = None
        non_ghost_parents = []
        for p in revobj.parent_ids:
            if p in self.excluded_revisions:
                continue
            try:
                parent_mark = self.revid_to_mark[p]
                non_ghost_parents.append(b':%s' % parent_mark)
            except KeyError:
                continue
        if non_ghost_parents:
            from_ = non_ghost_parents[0]
            merges = non_ghost_parents[1:]
        else:
            from_ = None
            merges = None
        if self.plain_format:
            properties = None
        else:
            properties = revobj.properties
            for prop in self.properties_to_exclude:
                try:
                    del properties[prop]
                except KeyError:
                    pass
        return commands.CommitCommand(git_ref, mark, author_info, committer_info, revobj.message.encode('utf-8'), from_, merges, file_cmds, more_authors=more_author_info, properties=properties)

    def _get_revision_trees(self, revids):
        missing = []
        by_revid = {}
        for revid in revids:
            if revid == breezy.revision.NULL_REVISION:
                by_revid[revid] = self.branch.repository.revision_tree(revid)
            elif revid not in self.tree_cache:
                missing.append(revid)
        for tree in self.branch.repository.revision_trees(missing):
            by_revid[tree.get_revision_id()] = tree
        for revid in revids:
            try:
                yield self.tree_cache[revid]
            except KeyError:
                yield by_revid[revid]
        for revid, tree in by_revid.items():
            self.tree_cache[revid] = tree

    def _get_filecommands(self, tree_old, tree_new):
        """Get the list of FileCommands for the changes between two revisions."""
        changes = tree_new.changes_from(tree_old)
        my_modified = list(changes.modified)
        file_cmds, rd_modifies, renamed = self._process_renames_and_deletes(changes.renamed, changes.removed, tree_new.get_revision_id(), tree_old)
        yield from file_cmds
        for change in changes.kind_changed:
            path = self._adjust_path_for_renames(change.path[0], renamed, tree_new.get_revision_id())
            my_modified.append(change)
        files_to_get = []
        for change in changes.added + changes.copied + my_modified + rd_modifies:
            if change.kind[1] == 'file':
                files_to_get.append((change.path[1], (change.path[1], helpers.kind_to_mode('file', change.executable[1]))))
            elif change.kind[1] == 'symlink':
                yield commands.FileModifyCommand(change.path[1].encode('utf-8'), helpers.kind_to_mode('symlink', False), None, tree_new.get_symlink_target(change.path[1]).encode('utf-8'))
            elif change.kind[1] == 'directory':
                if not self.plain_format:
                    yield commands.FileModifyCommand(change.path[1].encode('utf-8'), helpers.kind_to_mode('directory', False), None, None)
            else:
                self.warning("cannot export '%s' of kind %s yet - ignoring" % (change.path[1], change.kind[1]))
        for (path, mode), chunks in tree_new.iter_files_bytes(files_to_get):
            yield commands.FileModifyCommand(path.encode('utf-8'), mode, None, b''.join(chunks))

    def _process_renames_and_deletes(self, renames, deletes, revision_id, tree_old):
        file_cmds = []
        modifies = []
        renamed = []
        must_be_renamed = {}
        old_to_new = {}
        deleted_paths = {change.path[0] for change in deletes}
        for change in renames:
            emit = change.kind[1] != 'directory' or not self.plain_format
            if change.path[1] in deleted_paths:
                if emit:
                    file_cmds.append(commands.FileDeleteCommand(change.path[1].encode('utf-8')))
                deleted_paths.remove(change.path[1])
            if self.is_empty_dir(tree_old, change.path[0]):
                self.note('Skipping empty dir {} in rev {}'.format(change.path[0], revision_id))
                continue
            renamed.append(change.path)
            old_to_new[change.path[0]] = change.path[1]
            if emit:
                file_cmds.append(commands.FileRenameCommand(change.path[0].encode('utf-8'), change.path[1].encode('utf-8')))
            if change.changed_content or change.meta_modified():
                modifies.append(change)
            if change.kind == ('directory', 'directory'):
                for p, e in tree_old.iter_entries_by_dir(specific_files=[change.path[0]]):
                    if e.kind == 'directory' and self.plain_format:
                        continue
                    old_child_path = osutils.pathjoin(change.path[0], p)
                    new_child_path = osutils.pathjoin(change.path[1], p)
                    must_be_renamed[old_child_path] = new_child_path
        if must_be_renamed:
            renamed_already = set(old_to_new.keys())
            still_to_be_renamed = set(must_be_renamed.keys()) - renamed_already
            for old_child_path in sorted(still_to_be_renamed):
                new_child_path = must_be_renamed[old_child_path]
                if self.verbose:
                    self.note('implicitly renaming {} => {}'.format(old_child_path, new_child_path))
                file_cmds.append(commands.FileRenameCommand(old_child_path.encode('utf-8'), new_child_path.encode('utf-8')))
        for change in deletes:
            if change.path[0] not in deleted_paths:
                continue
            if change.kind[0] == 'directory' and self.plain_format:
                continue
            file_cmds.append(commands.FileDeleteCommand(change.path[0].encode('utf-8')))
        return (file_cmds, modifies, renamed)

    def _adjust_path_for_renames(self, path, renamed, revision_id):
        for old, new in renamed:
            if path == old:
                self.note('Changing path %s given rename to %s in revision %s' % (path, new, revision_id))
                path = new
            elif path.startswith(old + '/'):
                self.note('Adjusting path %s given rename of %s to %s in revision %s' % (path, old, new, revision_id))
                path = path.replace(old + '/', new + '/')
        return path

    def emit_tags(self):
        for tag, revid in self.branch.tags.get_tag_dict().items():
            try:
                mark = self.revid_to_mark[revid]
            except KeyError:
                self.warning('not creating tag %r pointing to non-existent revision %s' % (tag, revid))
            else:
                git_ref = b'refs/tags/%s' % tag.encode('utf-8')
                if self.plain_format and (not check_ref_format(git_ref)):
                    if self.rewrite_tags:
                        new_ref = sanitize_ref_name_for_git(git_ref)
                        self.warning('tag %r is exported as %r to be valid in git.', git_ref, new_ref)
                        git_ref = new_ref
                    else:
                        self.warning('not creating tag %r as its name would not be valid in git.', git_ref)
                        continue
                self.print_cmd(commands.ResetCommand(git_ref, b':%s' % mark))