from .... import errors
from .... import transport as _mod_transport
from .... import ui
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....textfile import text_file
from ....timestamp import format_highres_date
from ....trace import mutter
from ...testament import StrictTestament
from ..bundle_data import BundleInfo, RevisionInfo
from . import BundleSerializer, _get_bundle_header, binary_diff
class BundleSerializerV08(BundleSerializer):

    def read(self, f):
        """Read the rest of the bundles from the supplied file.

        :param f: The file to read from
        :return: A list of bundles
        """
        return BundleReader(f).info

    def check_compatible(self):
        if self.source.supports_rich_root():
            raise errors.IncompatibleBundleFormat('0.8', repr(self.source))

    def write(self, source, revision_ids, forced_bases, f):
        """Write the bundless to the supplied files.

        :param source: A source for revision information
        :param revision_ids: The list of revision ids to serialize
        :param forced_bases: A dict of revision -> base that overrides default
        :param f: The file to output to
        """
        self.source = source
        self.revision_ids = revision_ids
        self.forced_bases = forced_bases
        self.to_file = f
        self.check_compatible()
        with source.lock_read():
            self._write_main_header()
            with ui.ui_factory.nested_progress_bar() as pb:
                self._write_revisions(pb)

    def write_bundle(self, repository, revision_id, base_revision_id, out):
        """Helper function for translating write_bundle to write"""
        forced_bases = {revision_id: base_revision_id}
        if base_revision_id is NULL_REVISION:
            base_revision_id = None
        graph = repository.get_graph()
        revision_ids = graph.find_unique_ancestors(revision_id, [base_revision_id])
        revision_ids = list(repository.get_graph().iter_topo_order(revision_ids))
        revision_ids.reverse()
        self.write(repository, revision_ids, forced_bases, out)
        return revision_ids

    def _write_main_header(self):
        """Write the header for the changes"""
        f = self.to_file
        f.write(_get_bundle_header('0.8'))
        f.write(b'#\n')

    def _write(self, key, value, indent=1, trailing_space_when_empty=False):
        """Write out meta information, with proper indenting, etc.

        :param trailing_space_when_empty: To work around a bug in earlier
            bundle readers, when writing an empty property, we use "prop: 
"
            rather than writing "prop:
".
            If this parameter is True, and value is the empty string, we will
            write an extra space.
        """
        if indent < 1:
            raise ValueError('indentation must be greater than 0')
        f = self.to_file
        f.write(b'#' + b' ' * indent)
        f.write(key.encode('utf-8'))
        if not value:
            if trailing_space_when_empty and value == '':
                f.write(b': \n')
            else:
                f.write(b':\n')
        elif isinstance(value, bytes):
            f.write(b': ')
            f.write(value)
            f.write(b'\n')
        elif isinstance(value, str):
            f.write(b': ')
            f.write(value.encode('utf-8'))
            f.write(b'\n')
        else:
            f.write(b':\n')
            for entry in value:
                f.write(b'#' + b' ' * (indent + 2))
                if isinstance(entry, bytes):
                    f.write(entry)
                else:
                    f.write(entry.encode('utf-8'))
                f.write(b'\n')

    def _write_revisions(self, pb):
        """Write the information for all of the revisions."""
        last_rev_id = None
        last_rev_tree = None
        i_max = len(self.revision_ids)
        for i, rev_id in enumerate(self.revision_ids):
            pb.update('Generating revision data', i, i_max)
            rev = self.source.get_revision(rev_id)
            if rev_id == last_rev_id:
                rev_tree = last_rev_tree
            else:
                rev_tree = self.source.revision_tree(rev_id)
            if rev_id in self.forced_bases:
                explicit_base = True
                base_id = self.forced_bases[rev_id]
                if base_id is None:
                    base_id = NULL_REVISION
            else:
                explicit_base = False
                if rev.parent_ids:
                    base_id = rev.parent_ids[-1]
                else:
                    base_id = NULL_REVISION
            if base_id == last_rev_id:
                base_tree = last_rev_tree
            else:
                base_tree = self.source.revision_tree(base_id)
            force_binary = i != 0
            self._write_revision(rev, rev_tree, base_id, base_tree, explicit_base, force_binary)
            last_rev_id = base_id
            last_rev_tree = base_tree

    def _testament_sha1(self, revision_id):
        return StrictTestament.from_revision(self.source, revision_id).as_sha1()

    def _write_revision(self, rev, rev_tree, base_rev, base_tree, explicit_base, force_binary):
        """Write out the information for a revision."""

        def w(key, value):
            self._write(key, value, indent=1)
        w('message', rev.message.split('\n'))
        w('committer', rev.committer)
        w('date', format_highres_date(rev.timestamp, rev.timezone))
        self.to_file.write(b'\n')
        self._write_delta(rev_tree, base_tree, rev.revision_id, force_binary)
        w('revision id', rev.revision_id)
        w('sha1', self._testament_sha1(rev.revision_id))
        w('inventory sha1', rev.inventory_sha1)
        if rev.parent_ids:
            w('parent ids', rev.parent_ids)
        if explicit_base:
            w('base id', base_rev)
        if rev.properties:
            self._write('properties', None, indent=1)
            for name, value in sorted(rev.properties.items()):
                self._write(name, value, indent=3, trailing_space_when_empty=True)
        self.to_file.write(b'\n')

    def _write_action(self, name, parameters, properties=None):
        if properties is None:
            properties = []
        p_texts = ['%s:%s' % v for v in properties]
        self.to_file.write(b'=== ')
        self.to_file.write(' '.join([name] + parameters).encode('utf-8'))
        self.to_file.write(' // '.join(p_texts).encode('utf-8'))
        self.to_file.write(b'\n')

    def _write_delta(self, new_tree, old_tree, default_revision_id, force_binary):
        """Write out the changes between the trees."""
        DEVNULL = '/dev/null'
        old_label = ''
        new_label = ''

        def do_diff(file_id, old_path, new_path, action, force_binary):

            def tree_lines(tree, path, require_text=False):
                try:
                    tree_file = tree.get_file(path)
                except _mod_transport.NoSuchFile:
                    return []
                else:
                    if require_text is True:
                        tree_file = text_file(tree_file)
                    return tree_file.readlines()
            try:
                if force_binary:
                    raise errors.BinaryFile()
                old_lines = tree_lines(old_tree, old_path, require_text=True)
                new_lines = tree_lines(new_tree, new_path, require_text=True)
                action.write(self.to_file)
                internal_diff(old_path, old_lines, new_path, new_lines, self.to_file)
            except errors.BinaryFile:
                old_lines = tree_lines(old_tree, old_path, require_text=False)
                new_lines = tree_lines(new_tree, new_path, require_text=False)
                action.add_property('encoding', 'base64')
                action.write(self.to_file)
                binary_diff(old_path, old_lines, new_path, new_lines, self.to_file)

        def finish_action(action, file_id, kind, meta_modified, text_modified, old_path, new_path):
            entry = new_tree.root_inventory.get_entry(file_id)
            if entry.revision != default_revision_id:
                action.add_utf8_property('last-changed', entry.revision)
            if meta_modified:
                action.add_bool_property('executable', entry.executable)
            if text_modified and kind == 'symlink':
                action.add_property('target', entry.symlink_target)
            if text_modified and kind == 'file':
                do_diff(file_id, old_path, new_path, action, force_binary)
            else:
                action.write(self.to_file)
        delta = new_tree.changes_from(old_tree, want_unchanged=True, include_root=True)
        for change in delta.removed:
            action = Action('removed', [change.kind[0], change.path[0]]).write(self.to_file)
        for change in delta.added + delta.copied:
            action = Action('added', [change.kind[1], change.path[1]], [('file-id', change.file_id.decode('utf-8'))])
            meta_modified = change.kind[1] == 'file' and change.executable[1]
            finish_action(action, change.file_id, change.kind[1], meta_modified, change.changed_content, DEVNULL, change.path[1])
        for change in delta.renamed:
            action = Action('renamed', [change.kind[1], change.path[0]], [(change.path[1],)])
            finish_action(action, change.file_id, change.kind[1], change.meta_modified(), change.changed_content, change.path[0], change.path[1])
        for change in delta.modified:
            action = Action('modified', [change.kind[1], change.path[1]])
            finish_action(action, change.file_id, change.kind[1], change.meta_modified(), change.changed_content, change.path[0], change.path[1])
        for change in delta.unchanged:
            new_rev = new_tree.get_file_revision(change.path[1])
            if new_rev is None:
                continue
            old_rev = old_tree.get_file_revision(change.path[0])
            if new_rev != old_rev:
                action = Action('modified', [change.kind[1], change.path[1]])
                action.add_utf8_property('last-changed', new_rev)
                action.write(self.to_file)